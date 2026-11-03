from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CourseOverride(BaseModel):
    """Reviewer corrections for one course, applied after parsing.

    Item ids are those of the uncorrected evidence, and moved items keep them.
    """

    model_config = ConfigDict(extra="forbid")

    prerequisites: list[str] = Field(default_factory=list, description="items moved to the prerequisites")
    contents: list[str] = Field(default_factory=list, description="prerequisites moved back to the contents")
    drop: list[str] = Field(default_factory=list, description="items removed as not course material")
    laboratory: bool | None = None
    note: str = ""


class OverridesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    courses: dict[str, CourseOverride] = Field(default_factory=dict)
