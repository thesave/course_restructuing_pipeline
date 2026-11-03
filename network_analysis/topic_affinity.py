"""Topic affinity as consensus co-assignment over many community detections.

Each run drops a random fraction of the current courses, builds the topic
co-teaching graph from the remaining ones and, optionally, fuses it with a
k-nearest-neighbour graph of description embeddings; weighted Louvain then
finds communities at each resolution. The affinity of two topics is the
fraction of detections, among those in which both are present, that put them
in the same community. Pairs never put together are left out of the affinity
file, which the modules engine reads as unrelated.

    .venv/bin/python -m network_analysis.topic_affinity \
        data/restructuring/attempt-.../topics-global.yml --output DIR \
        [--embeddings topic-embeddings.npz --description-share 0.5] [--course ID ...]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from modules_engine.io import atomic_write_text, dump_yaml
from network_analysis.topic_embeddings import load_embeddings
from network_analysis.topic_graph import load_course_topics


WEIGHTINGS = ("jaccard", "newman", "count")


@dataclass(frozen=True)
class ConsensusConfig:
    resolutions: tuple[float, ...] = (1.0, 3.0, 10.0)
    runs: int = 20
    # Fraction of current courses dropped in each run (course resampling).
    drop_fraction: float = 0.1
    # Weight of the description layer in the fused graph: 0 is co-teaching
    # only, 1 is descriptions only.
    description_share: float = 0.0
    neighbours: int = 10
    weighting: str = "jaccard"
    seed: int = 0


def coteaching_weights(course_topics: dict[str, set[str]], weighting: str) -> dict[tuple[str, str], float]:
    """Edge weights between topics taught by a common course.

    ``count``: number of shared courses. ``newman``: each course adds
    1/(n-1) to the pairs among its n topics, so broad courses weigh less.
    ``jaccard``: shared courses over courses teaching either topic, so
    topics taught everywhere weigh less.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting}; expected one of {WEIGHTINGS}")
    weights: dict[tuple[str, str], float] = defaultdict(float)
    courses_of: dict[str, int] = defaultdict(int)
    for topics in course_topics.values():
        for topic in topics:
            courses_of[topic] += 1
        share = 1.0 / (len(topics) - 1) if weighting == "newman" and len(topics) > 1 else 1.0
        for pair in itertools.combinations(sorted(topics), 2):
            weights[pair] += share
    if weighting == "jaccard":
        return {(a, b): shared / (courses_of[a] + courses_of[b] - shared) for (a, b), shared in weights.items()}
    return dict(weights)


def description_neighbours(keys: list[str], vectors: np.ndarray, neighbours: int) -> dict[tuple[str, str], float]:
    """Each topic linked to its most similar descriptions (cosine > 0)."""
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    edges: dict[tuple[str, str], float] = {}
    for index, row in enumerate(similarity):
        for other in np.argsort(row)[::-1][:neighbours]:
            if row[other] > 0:
                pair = tuple(sorted((keys[index], keys[other])))
                edges[pair] = max(edges.get(pair, 0.0), float(row[other]))
    return edges


def fused_graph(
    topics: set[str],
    coteaching: dict[tuple[str, str], float],
    description: dict[tuple[str, str], float],
    share: float,
) -> nx.Graph:
    """Sum of both layers, each scaled to mean edge weight 1 and weighted by its share."""
    graph = nx.Graph()
    graph.add_nodes_from(topics)
    for layer, weight in ((coteaching, 1.0 - share), (description, share)):
        if not layer or weight <= 0:
            continue
        mean = statistics.fmean(layer.values())
        for (a, b), value in layer.items():
            if a in topics and b in topics:
                previous = graph.edges[a, b]["weight"] if graph.has_edge(a, b) else 0.0
                graph.add_edge(a, b, weight=previous + weight * value / mean)
    return graph


@dataclass(frozen=True)
class Consensus:
    topics: list[str]
    # Detections putting each pair together, and detections with both present.
    together: np.ndarray
    present: np.ndarray

    def frequencies(self) -> np.ndarray:
        frequency = np.divide(self.together, self.present, out=np.zeros_like(self.together), where=self.present > 0)
        np.fill_diagonal(frequency, 0.0)
        return frequency


def consensus(
    course_topics: dict[str, set[str]],
    embeddings: tuple[list[str], np.ndarray] | None,
    config: ConsensusConfig,
) -> Consensus:
    if config.description_share > 0 and embeddings is None:
        raise ValueError("a description share needs embeddings")
    all_topics = sorted(set().union(*course_topics.values()) | (set(embeddings[0]) if embeddings else set()))
    index = {topic: position for position, topic in enumerate(all_topics)}
    size = len(all_topics)
    together = np.zeros((size, size), dtype=np.float32)
    present = np.zeros((size, size), dtype=np.float32)
    description = (
        description_neighbours(*embeddings, config.neighbours)
        if embeddings is not None and config.description_share > 0
        else {}
    )
    courses = sorted(course_topics)
    keep = len(courses) - round(config.drop_fraction * len(courses))
    for run in range(config.runs):
        generator = random.Random(config.seed + run)
        kept = {course: course_topics[course] for course in generator.sample(courses, keep)}
        topics = set().union(*kept.values())
        if config.description_share > 0:
            # Descriptions keep every topic in the graph.
            topics |= set(embeddings[0])
        if config.description_share >= 1:
            coteaching = {}
        else:
            coteaching = coteaching_weights(kept, config.weighting)
        graph = fused_graph(topics, coteaching, description, config.description_share)
        mask = np.zeros(size, dtype=np.float32)
        mask[[index[topic] for topic in topics]] = 1.0
        for resolution in config.resolutions:
            communities = nx.community.louvain_communities(
                graph, weight="weight", resolution=resolution, seed=config.seed + run
            )
            membership = np.zeros((size, len(communities)), dtype=np.float32)
            for number, members in enumerate(communities):
                membership[[index[topic] for topic in members], number] = 1.0
            together += membership @ membership.T
            present += np.outer(mask, mask)
    return Consensus(all_topics, together, present)


def consensus_partition(result: Consensus, threshold: float = 0.5, seed: int = 0) -> dict[str, int]:
    """Communities of the graph of pairs together in at least ``threshold`` of detections."""
    frequency = result.frequencies()
    graph = nx.Graph()
    graph.add_nodes_from(result.topics)
    rows, columns = np.nonzero(np.triu(frequency >= threshold, 1))
    graph.add_weighted_edges_from(
        (result.topics[row], result.topics[column], float(frequency[row, column]))
        for row, column in zip(rows, columns)
    )
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    ordered = sorted(communities, key=lambda members: (-len(members), min(members)))
    return {topic: number for number, members in enumerate(ordered, start=1) for topic in members}


def topic_roles(course_topics: dict[str, set[str]], partition: dict[str, int]) -> dict[str, dict[str, float]]:
    """Participation coefficient and within-community degree z-score (Guimera
    and Amaral) on the co-teaching graph weighted by shared courses.

    A high participation coefficient marks a bridge topic whose co-teaching
    links spread over many communities.
    """
    weights = coteaching_weights(course_topics, "count")
    strength: dict[str, float] = defaultdict(float)
    by_community: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for (a, b), weight in weights.items():
        strength[a] += weight
        strength[b] += weight
        by_community[a][partition[b]] += weight
        by_community[b][partition[a]] += weight
    inside = {topic: by_community[topic][partition[topic]] for topic in partition}
    members: dict[int, list[str]] = defaultdict(list)
    for topic, community in partition.items():
        members[community].append(topic)
    roles = {}
    for topic, community in partition.items():
        total = strength.get(topic, 0.0)
        participation = 1 - sum((value / total) ** 2 for value in by_community[topic].values()) if total else 0.0
        peers = [inside[peer] for peer in members[community]]
        spread = statistics.pstdev(peers)
        roles[topic] = {
            "community": community,
            "participation": round(participation, 3),
            "within_degree_z": round((inside[topic] - statistics.fmean(peers)) / spread, 3) if spread else 0.0,
            "strength": round(total, 3),
        }
    return roles


def area_topics(course_topics: dict[str, set[str]], course_ids: tuple[str, ...]) -> set[str]:
    unknown = sorted(set(course_ids) - set(course_topics))
    if unknown:
        raise ValueError(f"Unknown course ids: {unknown}")
    return set().union(*(course_topics[course] for course in course_ids)) if course_ids else set().union(*course_topics.values())


def affinity_pairs(result: Consensus, topics: set[str]) -> list[list]:
    frequency = result.frequencies()
    selected = [position for position, topic in enumerate(result.topics) if topic in topics]
    pairs = []
    for row, column in itertools.combinations(selected, 2):
        score = round(float(frequency[row, column]), 3)
        if score > 0:
            pairs.append([result.topics[row], result.topics[column], score])
    return pairs


def save_consensus(path: pathlib.Path, result: Consensus, config: ConsensusConfig) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        topics=np.array(result.topics),
        together=result.together,
        present=result.present,
        config=np.array(repr(asdict(config))),
    )
    temporary.replace(path)


def load_consensus(path: pathlib.Path) -> Consensus:
    with np.load(path, allow_pickle=False) as data:
        return Consensus([str(topic) for topic in data["topics"]], data["together"], data["present"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topics_global", type=pathlib.Path, help="Path to an attempt's topics-global.yml")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output directory")
    parser.add_argument("--embeddings", type=pathlib.Path, help="topic-embeddings.npz for the description layer")
    parser.add_argument("--description-share", type=float, default=0.0)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--weighting", choices=WEIGHTINGS, default="jaccard")
    parser.add_argument("--resolutions", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--drop-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Co-assignment threshold of the consensus partition")
    parser.add_argument("--course", action="append", default=[], help="Restrict the affinity file to these courses' topics")
    args = parser.parse_args()
    if not 0 <= args.description_share <= 1:
        parser.error("--description-share must be between 0 and 1")

    _, shares = load_course_topics(args.topics_global)
    course_topics = {course: set(topics) for course, topics in shares.items() if topics}
    embeddings = None
    if args.embeddings is not None:
        keys, vectors, _ = load_embeddings(args.embeddings)
        embeddings = (keys, vectors)
    config = ConsensusConfig(
        resolutions=tuple(args.resolutions),
        runs=args.runs,
        drop_fraction=args.drop_fraction,
        description_share=args.description_share,
        neighbours=args.neighbours,
        weighting=args.weighting,
        seed=args.seed,
    )
    selected = area_topics(course_topics, tuple(args.course))
    result = consensus(course_topics, embeddings, config)
    partition = consensus_partition(result, args.threshold, args.seed)
    roles = topic_roles(course_topics, partition)

    args.output.mkdir(parents=True, exist_ok=True)
    save_consensus(args.output / "coassignment.npz", result, config)
    with (args.output / "consensus-partition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["topic", "community"])
        writer.writerows(sorted(partition.items()))
    atomic_write_text(args.output / "affinity.yml", dump_yaml({
        "source": f"consensus co-assignment over {args.topics_global}: {asdict(config)}",
        "pairs": affinity_pairs(result, selected),
    }, flow_leaves=True))
    atomic_write_text(args.output / "topic-roles.yml", dump_yaml({
        "description": "participation: share of co-teaching links outside the topic's consensus community "
        "(1 - sum of squared shares per community); within_degree_z: links inside it, as a z-score",
        "topics": dict(sorted(
            ((topic, roles[topic]) for topic in selected if topic in roles),
            key=lambda item: -item[1]["participation"],
        )),
    }))
    for name in ("coassignment.npz", "consensus-partition.csv", "affinity.yml", "topic-roles.yml"):
        print(args.output / name)


if __name__ == "__main__":
    main()
