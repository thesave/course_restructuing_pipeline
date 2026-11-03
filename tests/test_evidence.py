from __future__ import annotations

import pathlib
import tempfile
import unittest

import yaml

from evidence.cli import build_parser
from evidence.models import CourseOverride
from evidence.prepare import (
    apply_override,
    course_evidence,
    course_facts,
    dump_evidence,
    prepare_evidence,
    shared_syllabi,
)
from evidence.text import Item, parse_section, split_sentences


def tree(items: list[Item]) -> list:
    """Items as nested (text, children) pairs, children omitted for leaves."""
    return [(item.text, tree(item.items)) if item.items else item.text for item in items]


def texts(items: list[Item]) -> list[str]:
    return [item.text for item in items]


def course(contents: str = "", outcomes: str | list = "Outcomes.", **extra) -> dict:
    payload = {
        "course_title": {"id": "1", "name": "A course"},
        "credits": [6],
        "ssds": ["INF/01"],
        "syllabus": {
            "en": {
                "contents": {
                    "Learning outcomes": outcomes,
                    "Course contents": contents,
                    "Readings/Bibliography": "- Book one.  \n\n\n- Book two.",
                    "Teaching methods": "Lectures.\n\nLaboratory exercises in Python.",
                    "Office hours": "See the website.",
                }
            }
        },
    }
    payload.update(extra)
    return payload


class ParseSectionTest(unittest.TestCase):
    def test_module_headings_hold_plain_lines(self) -> None:
        parsed = parse_section("Module 1:\n\nScheduling\n\nFile System\n\nModule 2:\n\nC language", "en")
        self.assertEqual(
            tree(parsed.items),
            [("Module 1", ["Scheduling", "File System"]), ("Module 2", ["C language"])],
        )
        self.assertTrue(parsed.items[0].heading)

    def test_requirements_block_and_numbered_groups(self) -> None:
        text = (
            "Requirements\n\nA prior knowledge of algorithms is required to attend with profit this course.\n\n"
            "Course Contents\n\n1. The lifecycle of software systems:\n\n- phases and activities;\n- design.\n\n"
            "1. The UML language:\n\n- use case diagrams;\n\n1. Basics of user interface design"
        )
        parsed = parse_section(text, "en")
        self.assertEqual(texts(parsed.prerequisites), ["A prior knowledge of algorithms is required to attend with profit this course."])
        self.assertEqual(
            tree(parsed.items),
            [
                ("The lifecycle of software systems", ["phases and activities;", "design."]),
                ("The UML language", ["use case diagrams;"]),
                "Basics of user interface design",
            ],
        )

    def test_glued_bold_heading_plain_groups_and_italian_lines(self) -> None:
        text = (
            "**Module 1 (in person)**Introduction to spreadsheets\nOrganizing data in Excel\n\n"
            "Databases:\n\nLe chiavi\nCreazione di tabelle e query"
        )
        parsed = parse_section(text, "en")
        self.assertEqual(
            tree(parsed.items),
            [
                (
                    "Module 1 (in person)",
                    [
                        "Introduction to spreadsheets",
                        "Organizing data in Excel",
                        ("Databases", ["Le chiavi", "Creazione di tabelle e query"]),
                    ],
                )
            ],
        )
        self.assertEqual([item.language for item in parsed.items[0].items[2].items], ["it", "it"])
        self.assertEqual(parsed.items[0].items[0].language, "en")

    def test_formal_note_is_removed_and_gives_equivalent_courses(self) -> None:
        text = (
            "IMPORTANT FORMAL NOTE\n\nTHE COURSES\n\n- 88324 - System Administration T\n- B0832 - AMMINISTRAZIONE DI SISTEMI T\n\n"
            "ARE THE SAME COURSE, SHOWN WITH DIFFERENT CODES FOR FORMAL REASONS; THEY CANNOT BE CHOSEN TOGETHER\n\n"
            "Entry requirements: IN ORDER TO FOLLOW THE COURSE, OPERATING SYSTEMS ARE ESSENTIAL. LINUX IS NEEDED.\n\n"
            "Foundations\n\n- Security properties"
        )
        parsed = parse_section(text, "en")
        self.assertEqual(parsed.equivalent_courses, ["88324", "B0832"])
        self.assertEqual({removed.reason for removed in parsed.removed}, {"formal note"})
        self.assertEqual(len(parsed.prerequisites), 1)
        self.assertIn("LINUX IS NEEDED", parsed.prerequisites[0].text)
        self.assertEqual(tree(parsed.items), [("Foundations", ["Security properties"])])

    def test_hard_wrapped_lines_are_joined(self) -> None:
        text = (
            "Moreover, seminars on specific Artificial Intelligence topics are planned. This course is\n"
            "preparatory for the course of\nIntelligent Systems.\n\n"
            "Prerequisites: attending the course requires a medium\nknowledge of a high level programming language.\n\n"
            "- Introduction to Artificial Intelligence: brief history of AI,\n  main application fields."
        )
        parsed = parse_section(text, "en")
        self.assertEqual(
            texts(parsed.items),
            [
                "Moreover, seminars on specific Artificial Intelligence topics are planned.",
                "This course is preparatory for the course of Intelligent Systems.",
                "Introduction to Artificial Intelligence: brief history of AI, main application fields.",
            ],
        )
        self.assertEqual(
            texts(parsed.prerequisites),
            ["Prerequisites: attending the course requires a medium knowledge of a high level programming language."],
        )

    def test_contents_lead_ends_a_prerequisite_block(self) -> None:
        text = (
            "Requirements/Prior knowledge\n\nKnowledge of the relational model is required.\n\n"
            "Course contents: BI module\n\n- the BI pyramid\n- Data Visualizations workshop\n  \n  1. Data Viz fundamentals\n"
            "- Case study"
        )
        parsed = parse_section(text, "en")
        self.assertEqual(texts(parsed.prerequisites), ["Knowledge of the relational model is required."])
        self.assertEqual(
            tree(parsed.items),
            [("BI module", ["the BI pyramid", ("Data Visualizations workshop", ["Data Viz fundamentals"]), "Case study"])],
        )

    def test_standalone_title_ends_a_prerequisite_block(self) -> None:
        text = (
            "Pre-requisites\n\nStudents are expected to have:\n\n- Good programming skills.\n\n"
            "Course Objectives\n\nThe course covers vehicular systems."
        )
        parsed = parse_section(text, "en")
        self.assertEqual(tree(parsed.prerequisites), [("Students are expected to have", ["Good programming skills."])])
        self.assertEqual(tree(parsed.items), [("Course Objectives", ["The course covers vehicular systems."])])

    def test_course_material_mentioning_knowledge_is_not_a_prerequisite(self) -> None:
        text = (
            "No prior knowledge of computer science is required. Then, using the C language, which is explained "
            "without assuming any prior knowledge, the course analyses classic problems. At the end, the student "
            "is expected to know the main middleware."
        )
        parsed = parse_section(text, "en")
        self.assertEqual(texts(parsed.prerequisites), ["No prior knowledge of computer science is required."])
        self.assertEqual(len(parsed.items), 2)

    def test_inline_enumeration_and_administrative_sentences(self) -> None:
        text = (
            "1: Introduction to operating systems (OS) 2: Processes and threads 3: CPU Scheduling\n\n"
            "See the course main web page for more details.\n\n"
            "- Software validation. Students with specific learning disorders should contact the office."
        )
        parsed = parse_section(text, "en")
        self.assertEqual(
            texts(parsed.items),
            ["Introduction to operating systems (OS)", "Processes and threads", "CPU Scheduling", "Software validation."],
        )
        self.assertEqual([removed.reason for removed in parsed.removed], ["pointer", "accessibility"])

    def test_instructors_and_links_are_removed(self) -> None:
        text = (
            "First module, first semester; instructor: Ada Lovelace\n\nhttp://www.example.org/~ada/\n\n"
            "Syntax (BNF).\n\n------------ Modulo II ISI: Text Mining (Prof. Alan Turing)\n\n1. Text Representation"
        )
        parsed = parse_section(text, "en")
        self.assertEqual(
            tree(parsed.items),
            [("First module, first semester", ["Syntax (BNF)."]), ("Modulo II ISI: Text Mining", ["Text Representation"])],
        )
        self.assertEqual([removed.reason for removed in parsed.removed], ["instructor", "link", "instructor"])

    def test_inline_bullets_in_outcomes(self) -> None:
        parsed = parse_section("At the end of the course the student: - knows sorting; - knows P and NP.", "en")
        self.assertEqual(tree(parsed.items), [("At the end of the course the student", ["knows sorting;", "knows P and NP."])])

    def test_sentences_keep_abbreviations(self) -> None:
        self.assertEqual(
            split_sentences("Read M. van Steen, e.g. chapter 2. Then Prof. Rossi explains it. Done."),
            ["Read M. van Steen, e.g. chapter 2.", "Then Prof. Rossi explains it.", "Done."],
        )


class PrepareTest(unittest.TestCase):
    def test_service_fact_counts_disi_teachings_once(self) -> None:
        module = {"teaching_id": "434708", "ssd": "ING-INF/05", "credits": 9}
        payload = course(
            teachers=[
                {"name": "A", "affiliation": "disi", "modules": [module]},
                {"name": "B", "affiliation": "disi", "modules": [module]},
            ],
            programmes=[{"code": "9254", "department": "dei", "duration": 3, "name": {"en": "Electronics"}}],
        )
        facts = course_facts(payload)
        self.assertEqual(facts["service"], {"value": True, "disi_teaching_ids": ["434708"]})
        self.assertEqual(facts["programmes"][0]["cycle"], "first")
        self.assertEqual(facts["laboratory"], {"value": True, "evidence": ["Laboratory exercises in Python."]})
        payload["programmes"].append({"code": "6668", "department": "disi"})
        self.assertFalse(course_facts(payload)["service"]["value"])

    def test_evidence_numbers_items_and_merges_outcome_variants(self) -> None:
        payload = course(
            "Module 1\n\n- Sorting\n\nNo prior knowledge is required.",
            outcomes=["The student knows sorting.", "The student knows sorting.\n\nThe student knows graphs."],
        )
        evidence, notes = course_evidence("00819-A", payload)
        self.assertEqual(evidence["course"], "00819-A")
        self.assertEqual([item["text"] for item in evidence["outcomes"]], ["The student knows sorting.", "The student knows graphs."])
        self.assertEqual(notes["outcome_variants"], 2)
        self.assertEqual(evidence["contents"], [{"id": "c1", "text": "Module 1", "heading": True, "items": [{"id": "c1.1", "text": "Sorting"}]}])
        self.assertEqual(evidence["prerequisites"], [{"id": "p1", "text": "No prior knowledge is required."}])
        self.assertEqual(evidence["bibliography"], "- Book one.\n\n- Book two.")
        self.assertIn("bibliography: |-\n  - Book one.\n", dump_evidence(evidence))

    def test_italian_sections_are_the_fallback(self) -> None:
        payload = course("Sorting")
        payload["syllabus"]["en"]["contents"].pop("Learning outcomes")
        payload["syllabus"]["it"] = {"contents": {"Conoscenze e abilità da conseguire": "Lo studente conosce gli algoritmi."}}
        evidence, _ = course_evidence("1", payload)
        self.assertEqual(evidence["sources"], {"outcomes": "it", "contents": "en", "bibliography": "en"})
        self.assertNotIn("language", evidence["outcomes"][0])

    def test_overrides_move_drop_and_set_laboratory(self) -> None:
        evidence, _ = course_evidence("1", course("- Sorting\n- Basic Python is assumed\n- Room B, Tuesday"))
        apply_override(evidence, CourseOverride(prerequisites=["c2"], drop=["c3"], laboratory=False, note="checked"))
        self.assertEqual([item["id"] for item in evidence["contents"]], ["c1"])
        self.assertEqual(evidence["prerequisites"], [{"id": "c2", "text": "Basic Python is assumed"}])
        self.assertEqual(evidence["removed"], [{"text": "Room B, Tuesday", "reason": "review"}])
        self.assertEqual(evidence["facts"]["laboratory"]["reviewed"], True)
        self.assertEqual(evidence["review_note"], "checked")
        apply_override(evidence, CourseOverride(contents=["c2"]))
        self.assertEqual([item["id"] for item in evidence["contents"]], ["c1", "c2"])
        with self.assertRaisesRegex(ValueError, "no item c9"):
            apply_override(evidence, CourseOverride(drop=["c9"]))

    def test_shared_syllabi(self) -> None:
        first, _ = course_evidence("1", course("- Sorting"))
        second, _ = course_evidence("2", course("- sorting!"))
        third, _ = course_evidence("3", course("- Graphs"))
        self.assertEqual(shared_syllabi({"1": first, "2": second, "3": third}), {"1": ["2"], "2": ["1"]})

    def test_prepare_writes_evidence_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            courses = root / "courses"
            courses.mkdir()
            (courses / "course-00819-A.yml").write_text(yaml.safe_dump(course("- Sorting")), encoding="utf-8")
            (courses / "course-00819-B.yml").write_text(yaml.safe_dump(course("- Sorting")), encoding="utf-8")
            output = root / "evidence"
            overrides = output / "overrides.yml"
            output.mkdir()
            overrides.write_text("courses:\n  00819-B:\n    laboratory: false\n", encoding="utf-8")
            written = prepare_evidence(courses, output, overrides)
            self.assertEqual([path.name for path in written], ["evidence-00819-A.yml", "evidence-00819-B.yml", "report.md"])
            evidence = yaml.safe_load((output / "evidence-00819-A.yml").read_text(encoding="utf-8"))
            self.assertEqual(evidence["format"], "evidence/1")
            self.assertEqual(evidence["facts"]["same_syllabus_as"], ["00819-B"])
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Shared syllabi (1)\n", report)
            self.assertIn("## Laboratory (1)\n", report)
            overrides.write_text("courses:\n  '99999':\n    laboratory: false\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown courses: 99999"):
                prepare_evidence(courses, output, overrides)

    def test_cli_defaults(self) -> None:
        args = build_parser().parse_args(["prepare", "data/courses/2025", "data/evidence/2025"])
        self.assertIsNone(args.overrides)
        self.assertEqual(args.output_dir, pathlib.Path("data/evidence/2025"))


if __name__ == "__main__":
    unittest.main()
