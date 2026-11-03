from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TOPIC_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
CONSTRAINT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
MODULE_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$"


class TopicSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    description: str = ""
    ects: float = Field(gt=0)
    # Current courses teaching the topic; used only for reporting.
    courses: list[str] = Field(default_factory=list)


class CourseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = ""
    credits: float | None = None


class TopicsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    topics: dict[str, TopicSpec] = Field(min_length=1)
    courses: dict[str, CourseSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_keys(self) -> "TopicsFile":
        invalid = sorted(key for key in self.topics if re.fullmatch(TOPIC_KEY_PATTERN, key) is None)
        if invalid:
            raise ValueError(f"invalid topic keys: {invalid}")
        return self


class Dependency(BaseModel):
    """``requires`` must be taught before (or with) ``for``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    requires: str
    for_: str = Field(alias="for")
    source: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def reject_self_reference(self) -> "Dependency":
        if self.requires == self.for_:
            raise ValueError(f"a topic cannot depend on itself: {self.requires}")
        return self


class DependenciesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dependencies: list[Dependency] = Field(default_factory=list)


class AffinityFile(BaseModel):
    """Pairwise scores of how strongly two topics belong in the same module."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    pairs: list[tuple[str, str, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pairs(self) -> "AffinityFile":
        seen: set[frozenset[str]] = set()
        for first, second, score in self.pairs:
            if first == second:
                raise ValueError(f"an affinity pair needs two distinct topics: {first}")
            if score < 0:
                raise ValueError(f"affinity scores must be non-negative: {first}, {second}")
            pair = frozenset((first, second))
            if pair in seen:
                raise ValueError(f"duplicate affinity pair: {first}, {second}")
            seen.add(pair)
        return self


class Bounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "Bounds":
        if self.min is None and self.max is None:
            raise ValueError("bounds need min, max, or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot exceed max")
        return self


class Quantile(BaseModel):
    """A threshold taken from the data: this quantile of the affinity scores
    listed in the affinity file."""

    model_config = ConfigDict(extra="forbid")

    quantile: float = Field(ge=0, le=1)


Threshold = Annotated[float, Field(ge=0)] | Quantile


class RelatedOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Pairs scoring below this also count as unrelated.
    min_score: Threshold | None = None


class BoundHardness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: bool = True
    max: bool = True


CONSTRAINT_KINDS = ("together", "apart", "module_ects", "related_only")


class Constraint(BaseModel):
    """Exactly one of ``together``, ``apart``, ``module_ects`` or
    ``related_only`` (topics without an affinity pair may not share a module).

    A soft constraint may be violated at a cost of ``weight`` per unit of
    violation: per extra module a ``together`` group is split into, per extra
    topic sharing a module in an ``apart`` group, per ECTS outside the
    ``module_ects`` bounds, per unrelated pair sharing a module.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=CONSTRAINT_ID_PATTERN)
    together: list[str] | None = None
    apart: list[str] | None = None
    module_ects: Bounds | None = None
    related_only: RelatedOnly | None = None
    hard: bool | BoundHardness = True
    weight: float = Field(default=1.0, gt=0)
    rationale: str | None = None

    @field_validator("related_only", mode="before")
    @classmethod
    def accept_flag(cls, value: object) -> object:
        # `related_only: true` is shorthand for no minimum score.
        if value is True:
            return {}
        if value is False:
            raise ValueError("omit related_only instead of setting it to false")
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> "Constraint":
        kinds = [name for name in CONSTRAINT_KINDS if getattr(self, name) is not None]
        if len(kinds) != 1:
            raise ValueError(f"constraint {self.id} needs exactly one of {', '.join(CONSTRAINT_KINDS)}")
        topics = self.together if self.together is not None else self.apart
        if topics is not None:
            if len(topics) < 2:
                raise ValueError(f"constraint {self.id} needs at least two topics")
            if len(topics) != len(set(topics)):
                raise ValueError(f"constraint {self.id} repeats topics")
        if isinstance(self.hard, BoundHardness) and self.module_ects is None:
            raise ValueError(f"constraint {self.id}: per-bound hardness applies to module_ects only")
        return self

    @property
    def kind(self) -> str:
        return next(name for name in CONSTRAINT_KINDS if getattr(self, name) is not None)

    def is_hard(self, bound: Literal["min", "max"] | None = None) -> bool:
        if isinstance(self.hard, bool):
            return self.hard
        if bound is None:
            raise ValueError(f"constraint {self.id} has per-bound hardness")
        return getattr(self.hard, bound)


class ConstraintsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_constraints(self) -> "ConstraintsFile":
        ids = [item.id for item in self.constraints]
        duplicates = sorted({key for key in ids if ids.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate constraint ids: {duplicates}")
        for kind in ("module_ects", "related_only"):
            if sum(item.kind == kind for item in self.constraints) > 1:
                raise ValueError(f"at most one {kind} constraint is allowed")
        return self


TermName = Literal["soft_constraint_penalty", "cohesion", "module_count"]
# Terms the engine knows, with the direction in which each is optimised.
TERM_SENSES: dict[str, Literal["minimise", "maximise"]] = {
    "soft_constraint_penalty": "minimise",
    "cohesion": "maximise",
    "module_count": "minimise",
}


class ObjectiveTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: TermName
    # Cohesion only: each pair sharing a module counts its affinity minus the
    # baseline (pairs missing from the affinity file score 0), so weakly
    # related topics cost something to put together.
    baseline: Threshold | None = None
    # Coefficient in weighted mode; ignored in lexicographic mode.
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_baseline(self) -> "ObjectiveTerm":
        if self.baseline is not None and self.term != "cohesion":
            raise ValueError("only the cohesion term takes a baseline")
        return self


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["lexicographic", "weighted"] = "lexicographic"
    terms: list[ObjectiveTerm] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terms(self) -> "Objective":
        names = [item.term for item in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("objective terms must be unique")
        return self


class PrefixedModule(BaseModel):
    """A module fixed by human review: its topics share one module, apart
    from other pre-fixed modules, and (when exclusive) no other topic joins."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(pattern=MODULE_KEY_PATTERN)
    topics: list[str] = Field(min_length=1)
    exclusive: bool = True
    note: str | None = None

    @model_validator(mode="after")
    def validate_topics(self) -> "PrefixedModule":
        if re.fullmatch(r"m[0-9]+", self.key):
            raise ValueError(f"{self.key}: names like m01 are kept for unnamed modules")
        if len(self.topics) != len(set(self.topics)):
            raise ValueError(f"pre-fixed module {self.key} repeats topics")
        return self


class PrefixedModulesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    modules: list[PrefixedModule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_modules(self) -> "PrefixedModulesFile":
        keys = [item.key for item in self.modules]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate pre-fixed module keys: {duplicates}")
        topics = [topic for item in self.modules for topic in item.topics]
        shared = sorted({topic for topic in topics if topics.count(topic) > 1})
        if shared:
            raise ValueError(f"topics in several pre-fixed modules: {shared}")
        return self


@dataclass(frozen=True)
class Problem:
    topics: TopicsFile
    dependencies: tuple[Dependency, ...]
    constraints: tuple[Constraint, ...]
    objective: Objective
    affinity: AffinityFile | None
    prefixed: tuple[PrefixedModule, ...] = ()


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float = 60.0
    workers: int = 8
    seed: int = 0
    max_modules: int | None = None
