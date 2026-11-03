# First attempt (reference only)

This folder keeps the artefacts of the first attempt at module generation, made
in `unibo-courses-cs-restructuring` before the pipeline was forked. They are
not inputs of the new pipeline, which re-extracts topics from the course corpus
in `data/courses/`.

`programming-languages/` is the prototype area used to develop the module
engine: five courses (04138, 72940, 81609, 72671, 81672) with 81 topics, taken
from the LLM topic extraction of `data/restructuring/attempt-2026-10-06-18-00`
in the source project. It contains the engine inputs (`topics.yml`,
`affinity.yml` from co-teaching, `constraints.yml`, `objective.yml`, and
`dependencies.yml`, an unreviewed draft written by an LLM from the topic
descriptions), the last run (`run/modules.yml` and `run/modules.mmd`), and the
affinity comparison (`affinity-variants/`, with `evaluation.md` comparing
co-teaching, consensus and description-fused affinities). The run's recorded
input paths point into this folder, so it can still be reviewed with
`./review-modules.sh first-attempt/programming-languages/run`.
