# Project map

This repository implements the course restructuring pipeline designed by the
DISI service-teaching working group: from the syllabi of current courses to a
reviewed proposal of reusable modules and their dependencies. It was forked
from `unibo-courses-cs-restructuring` (the scraping, clustering and LLM
restructuring project), keeping only the parts written for the new pipeline.
The pipeline design, with the steps that exist and those still proposed, is in
`0.pipeline design/`:

```text
0.pipeline design/analysis-pipeline.mmd overall schema (render.sh -> pdf|png|drawio)
0.pipeline design/upstream-proposal.md  design of the courses -> topics stages
evidence/                               stage 1: citable syllabus items and course facts
network_analysis/                       topic affinity (co-teaching, embeddings, consensus)
modules_engine/                         constraint solver for modules, review page, rendering
data/courses/2025/                      course corpus (syllabi and course facts)
data/evidence/2025/                     stage 1 output, report and reviewer overrides
data/modules-engine/<area>/             engine inputs and runs (none yet)
first-attempt/                          artefacts of the first attempt, reference only
```

## Domain intent

The primary scope is first- and second-cycle teaching delivered outside DISI
degree programmes: a service course is one where DISI members deliver INF/01
or ING-INF/05 credits and none of the associated degree programmes belongs to
DISI. Internal DISI teaching is context (for affinity, for instance), not the
main target. Module proposals are discussion aids, not decisions: human review
must still account for prerequisites, depth, laboratories, assessment,
language, semester, campus, enrolment, regulation and organisational
feasibility.

## Status and provenance

- `data/courses/2025/course-*.yml` is the course corpus (293 courses),
  imported unchanged from the source project's canonical merged courses at its
  commit `bf1a5c5`; `data/courses/README.md` describes the fields. It is the
  input of the new upstream stages, which re-extract topics from the syllabi,
  so the topics of the source project's restructuring attempts are not
  imported. Of those stages, only evidence preparation (stage 1) exists so
  far. Until topic extraction exists, `modules_engine prepare` and
  `network_analysis` still read an attempt directory of the source project
  (`data/restructuring/attempt-*/topics-global.yml`), as a bridge.
- `first-attempt/programming-languages/` keeps the prototype area of the
  first attempt (five courses, 81 topics derived from the source project's
  `attempt-2026-10-06-18-00`), for reference only: its `dependencies.yml` is
  an unreviewed LLM draft (`source: llm`), and `affinity-variants/` with
  `evaluation.md` comes from the affinity comparison. New engine inputs and
  runs go under `data/modules-engine/<area>/`.
- Deliberately not imported: the description embeddings cache
  (`topic-embeddings.npz`, computed on the old topics), the clustering of
  courses and the manual decompositions of clusters 1, 5, 7 and 27. The
  pipeline works on topics, not course clusters, and the quality of each
  model-driven stage is judged by human review inside the pipeline rather
  than against external references.
- `network_analysis/topic_graph.py` was written in the source project by
  Alina Sirbu and is copied unchanged; the other modules reuse its loaders.

## Runtime and conventions

- Un-packaged Python repository: run commands from the repository root with
  `.venv/bin/python -m ...` so absolute package imports resolve. The `.venv`
  uses Python 3.12 (`requirements.txt`): OR-Tools has no wheels for 3.15 yet.
- Tests use `unittest`, not pytest:
  `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.
- Node is used for Mermaid only: after `npm install`, mermaid-cli renders
  diagrams, the review page inlines `node_modules/mermaid/dist/mermaid.min.js`,
  and `0.pipeline design/mermaid-to-drawio.mjs` uses Puppeteer.
  `mermaid-config.json` sets the classic look, SVG text labels (no HTML labels)
  and raised size limits; generated Mermaid labels carry no HTML markup.
- Use Conventional Commits with a meaningful scope (for example,
  `feat(modules_engine): ...`). Mark breaking changes with `!` and a
  `BREAKING CHANGE:` footer.
- Treat course IDs as opaque strings: leading zeroes and suffixes such as
  `00819-B` are meaningful.
- Preserve atomic writes for generated artefacts.

## `0.pipeline design/`

- `analysis-pipeline.mmd`: the overall schema. It uses only Mermaid 10 syntax
  with quoted labels, so editors bundling older Mermaid (such as VS Code's
  Mermaid Editor) can parse it. Dashed borders mark proposed steps.
- `render.sh [pdf|png|drawio]` renders every `.mmd` of the folder; `drawio`
  runs `mermaid-to-drawio.mjs`, which renders with Mermaid in headless Chrome
  and writes an editable draw.io file laid out like the PDF. Cell ids are the
  Mermaid node ids and styles record `mermaidType`/`mermaidClass`, so edits
  made in draw.io can be mapped back onto the `.mmd` (positions cannot: Mermaid
  stores none). Re-running `render.sh drawio` overwrites the `.drawio` file.
- `upstream-proposal.md`: the seven proposed stages from courses to topics
  (per-course extraction with evidence and attributes, canonicalisation, topic
  review, re-annotation, derived inputs, dependency proposal and review),
  quality gates, the human checks closing each model-driven stage, the pilot
  on a small set of service courses and the open questions for the group.

## `evidence/`

Stage 1 of the upstream design, without model calls.
`.venv/bin/python -m evidence prepare data/courses/2025 data/evidence/2025 [--overrides FILE]`
writes one `evidence-<course>.yml` per course (`format: evidence/1`; the course
id comes from the file name, so suffixes such as `-A` survive) and
`report.md`. Every run processes the whole corpus, because shared syllabi are
found across courses.

- Sections come from the English syllabus when present, else from the Italian
  one (`sources` records which). Learning outcomes and contents become item
  trees (`o1`, `c3`, `c3.2`: module headings, then groups such as "The UML
  language:", then leaves); paragraphs are split into sentences; hard-wrapped
  lines are joined; merged courses with several outcome texts keep each
  distinct item once. Items of an English section detected as Italian carry
  `language: it`. The bibliography is kept as plain text.
- `prerequisites` gathers prerequisite statements (blocks under headings such
  as "Prerequisites" or "Requirements/Prior knowledge", lead-ins such as
  "Students are expected to have:", and matching sentences), so that topic
  extraction does not read them as course material; outcome phrasing ("at the
  end ... is expected to know") is excluded. `removed` lists administrative
  remarks (instructors, links, accessibility, timetables, language of
  teaching, pointers to web pages, the "IMPORTANT FORMAL NOTE" blocks, whose
  codes become `facts.equivalent_courses`).
- `facts`: credits, sectors, languages, campi, schedules, teaching ids,
  teachers, programmes (department and cycle), `service` (the domain
  definition above, with the DISI teaching ids behind it), `laboratory` (a
  keyword flag on the teaching methods with the sentences behind it; UniBo's
  workplace-safety requirement counts), and `same_syllabus_as` (courses with
  identical outcomes and contents).
- Parsing is heuristic, so the stage ends with a human check: `report.md`
  lists prerequisites, removed text, prose contents, long items, untranslated
  passages, Italian sections, outcome variants, shared syllabi, equivalent
  courses and laboratory flags. Corrections go into `overrides.yml` next to
  the output (`courses: {ID: {prerequisites, contents, drop, laboratory,
  note}}`, items named by the ids of the uncorrected output; moved items keep
  their ids), which every run reapplies; unknown courses or ids are errors.
  When the parser changes, check that the ids named in the overrides still
  point at the same items.

Tests are in `tests/test_evidence.py`.

## `network_analysis/`

- `topic_graph.py`: topic co-occurrence graph (topics linked by shared
  current courses), Louvain communities by resolution or ECTS cap, tables,
  Sankey diagrams and GraphML. The GraphML `community` attribute holds only the
  last run's communities; use the `topic-communities-*.csv` tables.
- `topic_embeddings.py`: embeds topic descriptions (key + description) with
  `paraphrase-multilingual-mpnet-base-v2` into `topic-embeddings.npz`. It
  imports only numpy, PyYAML and sentence-transformers, so it can run in a
  separate environment (`requirements-embeddings.txt` explains the Intel Mac
  pins).
- `topic_affinity.py`: consensus co-assignment. Each run drops a fraction of
  current courses, builds the co-teaching graph (`jaccard`, `newman` or
  `count` weights), optionally fuses a description k-nearest-neighbour layer
  (`--description-share`, 0 = co-teaching only), and runs Louvain at each
  resolution; a pair's affinity is the share of detections putting it
  together. Writes an engine-ready `affinity.yml` (restricted to `--course`
  topics), `coassignment.npz`, a thresholded `consensus-partition.csv` and
  `topic-roles.yml` (participation coefficient: bridge topics).
- `affinity_evaluation.py`: compares partitions by course recovery (NMI with
  the owning course over single-course topics; share of courses whose
  exclusive topics are split), consensus instability, and Spearman agreement
  of induced course similarities with manual decompositions passed with
  `--reference` (rows mapped to course ids by the id in parentheses, else by a
  unique title); this repository ships no such references.
- On co-teaching alone, every course's exclusive topics stay in one community
  (they form a clique with no other links); only the description layer can
  split them.

Tests are in `tests/test_topic_affinity.py`.

## `modules_engine/`

A CP-SAT solver (OR-Tools) partitions topics into modules under explicit
constraints and derives module dependencies from topic dependencies.

- `.venv/bin/python -m modules_engine prepare ATTEMPT_DIR OUT_DIR [--course ID ...]`
  writes `topics.yml` (topic `ects` from `topic_weights`, plus the selected
  current courses teaching it) and `affinity.yml` (Jaccard index of the sets of
  current courses teaching each topic pair, over the whole corpus).
- `.venv/bin/python -m modules_engine solve TOPICS DEPENDENCIES CONSTRAINTS OBJECTIVE [--affinity AFFINITY] [--review FILE | --prefixed FILE --hint MODULES_YML] --output DIR`
  writes `modules.yml` (modules, reduced module dependencies with the topic
  edges behind them, violations, metrics, per-course spread) and `modules.mmd`.
  `--time-limit` is per objective stage; a stage stopped by it only bounds its
  term, so the report gives both `value_at_stage` and the `final` value. With
  several workers and time limits, runs are not reproducible.
- Inputs are separate, hand-editable files. `dependencies.yml` lists
  `{requires, for, source}` edges and must be acyclic. `constraints.yml` lists
  `together`, `apart`, (at most one) `module_ects` and (at most one)
  `related_only` constraints, each hard or soft with a `weight` per unit of
  violation (per extra module, per extra co-located topic, per ECTS out of
  bounds, per co-located unrelated pair). `objective.yml` picks terms
  (`soft_constraint_penalty`, `cohesion`, `module_count`), `lexicographic`
  (default) or `weighted`. `affinity.yml` is needed by `cohesion` and
  `related_only`.
- A pair missing from `affinity.yml` scores 0. With a cohesion `baseline` (a
  number or `{quantile: q}` of the listed scores) every co-located pair counts
  score minus baseline, so weak and missing pairs cost something.
  `related_only` treats missing pairs (and, with `min_score`, weak ones) as
  unrelated; with co-teaching affinity that means "never taught together".
- Module indices are a topological order (`index(requires) <= index(for)`), so
  the module graph is acyclic by construction. When hard constraints conflict,
  the report lists a shrunk set of conflicting ones
  (`conflicting_hard_constraints`).
- Human review loop: `./review-modules.sh [RUN_DIR]` (or
  `.venv/bin/python -m modules_engine.review [RUN_DIR] [--review FILE] [--open]`;
  without RUN_DIR, the most recent run under `data/modules-engine/` with
  modules) writes `RUN_DIR/review.html` and opens it. The page is
  self-contained (Mermaid and js-yaml inlined from `node_modules`), gitignored
  and regenerated on demand; it starts from the review file
  `RUN_DIR/review.yml` (or `--review FILE`) when that was saved for the same
  proposal (matching `fingerprint`), otherwise from the solver's proposal.
  Topics are dragged between modules (or selected and clicked), new modules
  start from a dashed card, and modules are renamed and locked; the page
  re-evaluates ECTS, violations, cohesion and dependencies exactly as
  `report.py` and redraws a Mermaid preview.
- Review files (`format: modules-review/1`) are the artefact to keep and
  commit: the `modules` arrangement (`{key, locked, topics}`, block style,
  editable by hand), then the run's data (topics, dependencies, constraints
  with resolved thresholds, affinity, proposal) one item per line. The page's
  Open, Save and Save as work on them; Save overwrites the opened file where
  the browser has the File System Access API (Chrome, Edge) and downloads
  otherwise. Saving is always allowed; the Re-solve dialog is disabled only
  when locked modules form a dependency cycle (directly or through other
  topics). Topics a hand-edited file does not place are gathered in an
  unlocked `unassigned` module.
- `solve --review FILE` keeps a review's locked modules as exclusive pre-fixed
  modules and starts from its whole arrangement; it replaces `--prefixed FILE`
  (hand-written `modules: [{key, topics, exclusive}]`) and `--hint
  MODULES_YML`. Pre-fixed modules are hard: their topics share one module,
  distinct from other pre-fixed modules and (when exclusive) closed to other
  topics; they keep their key in the report (`mNN` names are reserved for
  unnamed modules). Known issue: re-solving after a small edit still degrades
  or reshuffles the rest of the proposal; a stability term that preserves
  reviewed groupings is the proposed fix.
- `.venv/bin/python -m modules_engine.render PATH... [--format pdf|png|svg]`
  renders `modules.mmd` files (or every one under a directory) next to their
  source.
- `report.py` re-evaluates the assignment independently of the solver
  variables; keep violations and penalties defined identically there, in the
  solver, and in `review_template.html`.
- The model has one boolean per topic and module, so it targets areas of about
  a hundred topics, not a whole corpus of thousands of topics at once.

Tests are in `tests/test_modules_engine.py`.
