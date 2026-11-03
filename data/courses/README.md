# Course corpus

One YAML file per course of academic year 2025 (`2025/course-<id>.yml`, 293
courses), imported unchanged from `unibo-courses-cs-restructuring`, where
`data/courses/.files/2025/` holds the canonical merged courses produced by its
scrapers (`scraping/download_teachings.py`, then `scraping/merge_teachings.py`).
The import corresponds to commit `bf1a5c5` of that repository (2026-07-15).
Its clustering embedding caches (`embeddings-*.yml` in the same folder) were not
imported.

Each file merges the teaching instances of one course: instances join when they
share a teaching id, the same set of degree programmes, the same syllabus
(learning outcomes aside) or similar learning outcomes, and distinct groups of
the same course get suffixes such as `00819-A`. Course ids are opaque strings.

The main fields are:

| Field | Content |
|---|---|
| `course_title` | `id` and `name` |
| `credits`, `ssds` | credits and scientific sectors (e.g. `INF/01`) of the course |
| `languages`, `teaching_modes`, `campi`, `schedules` | delivery facts |
| `teachers` | id, name, role, `affiliation` (department), sector, and the teaching modules each one delivers (teaching id, credits, sector, campus, language, schedule, syllabus URLs) |
| `programmes` | degree programmes using the course: code, name, campus, duration, access, `department` |
| `syllabus.it`, `syllabus.en` | title and `contents`, a mapping from section heading to text |

The syllabus sections present in nearly every course are the learning outcomes
(`Learning outcomes` / `Conoscenze e abilità da conseguire`), the contents
(`Course contents` / `Contenuti`), the bibliography, the teaching methods and
the assessment methods. No course has a prerequisites section.

Teachers' `affiliation` and programmes' `department` are what the
service-course definition needs: DISI members deliver INF/01 or ING-INF/05
credits and no associated degree programme belongs to DISI.
