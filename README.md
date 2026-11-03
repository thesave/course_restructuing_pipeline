# Course restructuring pipeline

Tools of the DISI service-teaching working group for turning the syllabi of
current courses into a reviewed proposal of reusable modules and their
dependencies. The overall design is in `0.pipeline design/` (schema and the
proposal for the upstream stages); `AGENTS.md` describes every component.

## Setup

```sh
# Python 3.12 (OR-Tools has no wheels for 3.15 yet)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Mermaid rendering, review page and draw.io export
npm install
```

Description embeddings need sentence-transformers in a separate environment;
see `requirements-embeddings.txt`.

## Quick start

```sh
# Tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Stage 1: evidence preparation (syllabus items, prerequisites, course facts);
# check data/evidence/2025/report.md and record corrections in overrides.yml there
.venv/bin/python -m evidence prepare data/courses/2025 data/evidence/2025

# Solve the first attempt's prototype area and review the proposal in the browser
D=first-attempt/programming-languages
.venv/bin/python -m modules_engine solve $D/topics.yml $D/dependencies.yml \
    $D/constraints.yml $D/objective.yml --affinity $D/affinity.yml \
    --output data/modules-engine/demo
./review-modules.sh      # reviews the most recent run under data/modules-engine/
# In the page, save the review as data/modules-engine/demo/review.yml, then
# re-solve keeping its locked modules (the page's Re-solve button shows the command):
.venv/bin/python -m modules_engine solve $D/topics.yml $D/dependencies.yml \
    $D/constraints.yml $D/objective.yml --affinity $D/affinity.yml \
    --review data/modules-engine/demo/review.yml --output data/modules-engine/demo-revised

# Render the pipeline schema (pdf, png or an editable drawio file)
"0.pipeline design/render.sh" drawio
```
