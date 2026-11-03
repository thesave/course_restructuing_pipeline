from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from network_analysis.affinity_evaluation import (
    course_recovery,
    load_partition,
    nmi,
    reference_agreement,
    reference_courses,
    stability,
)
from network_analysis.topic_affinity import (
    ConsensusConfig,
    affinity_pairs,
    coteaching_weights,
    consensus,
    consensus_partition,
    description_neighbours,
    load_consensus,
    save_consensus,
    topic_roles,
)
from network_analysis.topic_embeddings import load_embeddings, save_embeddings, topic_text


# Two groups of courses that share no topic, plus one course bridging them.
COURSES = {
    "c1": {"a", "b", "c"},
    "c2": {"a", "b", "c"},
    "c3": {"x", "y", "z"},
    "c4": {"x", "y", "z"},
}


class WeightsTest(unittest.TestCase):
    def test_coteaching_weightings(self) -> None:
        courses = {"c1": {"a", "b", "c"}, "c2": {"a", "b"}}
        self.assertEqual(coteaching_weights(courses, "count"), {("a", "b"): 2, ("a", "c"): 1, ("b", "c"): 1})
        self.assertEqual(coteaching_weights(courses, "newman"), {("a", "b"): 1.5, ("a", "c"): 0.5, ("b", "c"): 0.5})
        self.assertEqual(coteaching_weights(courses, "jaccard"), {("a", "b"): 1.0, ("a", "c"): 0.5, ("b", "c"): 0.5})
        with self.assertRaises(ValueError):
            coteaching_weights(courses, "pmi")

    def test_description_neighbours_keep_positive_nearest(self) -> None:
        vectors = np.array([[1, 0], [0.8, 0.6], [-1, 0]], dtype=np.float32)
        self.assertEqual(description_neighbours(["a", "b", "c"], vectors, 1), {("a", "b"): 0.800000011920929})


class ConsensusTest(unittest.TestCase):
    def test_disconnected_groups_never_share_a_community(self) -> None:
        result = consensus(COURSES, None, ConsensusConfig(resolutions=(1.0,), runs=3, drop_fraction=0.0))
        frequency = result.frequencies()
        index = {topic: position for position, topic in enumerate(result.topics)}
        self.assertEqual(frequency[index["a"], index["b"]], 1.0)
        self.assertEqual(frequency[index["a"], index["x"]], 0.0)
        partition = consensus_partition(result)
        self.assertEqual(partition["a"], partition["c"])
        self.assertNotEqual(partition["a"], partition["x"])
        self.assertEqual(affinity_pairs(result, {"a", "b", "x"}), [["a", "b", 1.0]])

    def test_descriptions_can_join_topics_never_taught_together(self) -> None:
        keys = ["a", "b", "c", "x", "y", "z"]
        # "c" and "x" have identical descriptions; the rest are orthogonal pairs.
        vectors = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        only_descriptions = consensus(
            COURSES, (keys, vectors), ConsensusConfig(resolutions=(1.0,), runs=1, description_share=1.0, neighbours=1)
        )
        index = {topic: position for position, topic in enumerate(only_descriptions.topics)}
        self.assertEqual(only_descriptions.frequencies()[index["c"], index["x"]], 1.0)
        with self.assertRaisesRegex(ValueError, "needs embeddings"):
            consensus(COURSES, None, ConsensusConfig(description_share=0.5))

    def test_dropped_courses_count_only_runs_with_both_topics(self) -> None:
        courses = {"c1": {"a", "b"}, "c2": {"c", "d"}}
        result = consensus(courses, None, ConsensusConfig(resolutions=(1.0,), runs=4, drop_fraction=0.5))
        index = {topic: position for position, topic in enumerate(result.topics)}
        # Each run keeps one course, so a and c are never present together.
        self.assertEqual(result.present[index["a"], index["c"]], 0)
        self.assertEqual(result.frequencies()[index["a"], index["b"]], 1.0)

    def test_roles_flag_bridge_topics(self) -> None:
        courses = {**COURSES, "c5": {"a", "x"}}
        roles = topic_roles(courses, {"a": 1, "b": 1, "c": 1, "x": 2, "y": 2, "z": 2})
        self.assertGreater(roles["a"]["participation"], 0)
        self.assertEqual(roles["b"]["participation"], 0)
        # The bridging link leaves a's links inside its community unchanged.
        self.assertEqual(roles["a"]["within_degree_z"], roles["b"]["within_degree_z"])
        self.assertEqual((roles["a"]["strength"], roles["b"]["strength"]), (5.0, 4.0))

    def test_consensus_and_embeddings_round_trip(self) -> None:
        result = consensus(COURSES, None, ConsensusConfig(resolutions=(1.0,), runs=1, drop_fraction=0.0))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            save_consensus(root / "c.npz", result, ConsensusConfig())
            loaded = load_consensus(root / "c.npz")
            save_embeddings(root / "e.npz", ["a"], np.ones((1, 2), dtype=np.float32), "model")
            keys, vectors, model = load_embeddings(root / "e.npz")
        self.assertEqual(loaded.topics, result.topics)
        np.testing.assert_array_equal(loaded.together, result.together)
        self.assertEqual((keys, vectors.shape, model), (["a"], (1, 2), "model"))
        self.assertEqual(topic_text("lr_parsing", "LR methods."), "lr parsing: LR methods.")


class EvaluationTest(unittest.TestCase):
    def test_nmi_and_course_recovery(self) -> None:
        self.assertEqual(nmi({"a": 1, "b": 1, "c": 2}, {"a": "x", "b": "x", "c": "y"}), 1.0)
        shares = {"c1": {"a": 1.0, "b": 1.0, "s": 1.0}, "c2": {"c": 1.0, "d": 1.0, "s": 1.0}}
        recovery = course_recovery({"a": 1, "b": 2, "c": 3, "d": 3, "s": 1}, shares)
        # c1's exclusive topics a, b are split; c2's c, d are not.
        self.assertEqual(recovery["courses_split"], 0.5)

    def test_stability_is_zero_when_detections_agree(self) -> None:
        self.assertEqual(stability(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])), 0.0)
        self.assertEqual(stability(np.array([[0, 0.5], [0.5, 0]])), 1.0)

    def test_reference_mapping_and_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "cluster.csv"
            path.write_text(
                ",M1,M2,Total CFU\n"
                "\"Course One (9730-A)\",3,,3\n"
                "\"Course Two (servizio, Bologna)\",,3,3\n"
                "\"Course Three (B1)\",2,1,3\n"
                "\"Unknown (Z9)\",1,1,2\n",
                encoding="utf-8",
            )
            modules, vectors = reference_courses(path, {"09730-A": "Course One", "22": "Course Two", "B1": "Other title"})
        self.assertEqual(modules, ["M1", "M2"])
        self.assertEqual(list(vectors), ["09730-A", "22", "B1"])
        shares = {"09730-A": {"a": 3.0}, "22": {"b": 3.0}, "B1": {"a": 2.0, "b": 1.0}}
        self.assertEqual(reference_agreement({"a": 1, "b": 2}, shares, vectors), 1.0)
        self.assertEqual(reference_agreement(None, shares, vectors), 1.0)

    def test_load_partition_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "long.csv").write_text("topic,community\na,1\nb,2\n", encoding="utf-8")
            (root / "table.csv").write_text("community,size,estimated_ects,topics,old_courses\n1,2,1.0,a; b,x\n", encoding="utf-8")
            self.assertEqual(load_partition(root / "long.csv"), {"a": 1, "b": 2})
            self.assertEqual(load_partition(root / "table.csv"), {"a": 1, "b": 1})


if __name__ == "__main__":
    unittest.main()
