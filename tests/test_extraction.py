from __future__ import annotations

import http.server
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest

import yaml

from extraction.claude_cli import ClaudeCliEndpoint, flatten
from extraction.cli import build_parser, load_endpoints, make_endpoints
from extraction.manual import ingest_reply, write_requests
from extraction.client import Endpoint, EndpointError, RequestError, api_base, strip_thinking
from extraction.models import Completion, EndpointSpec, EndpointsFile, Extraction, RunConfig
from extraction.prompt import course_messages, names_technology, render_bibliography, response_schema, retry_message, validate_extraction
from extraction.run import Extractor, select_courses, syllabus_groups


def evidence(course: str = "1", credits: int = 2, same: list[str] | None = None, service: bool = False) -> dict:
    return {
        "course": course,
        "title": f"Course {course}",
        "facts": {
            "credits": [credits],
            "programmes": [{"name": "Physics", "cycle": "first"}],
            "service": {"value": service},
            "same_syllabus_as": same or [],
        },
        "outcomes": [{"id": "o1", "text": "The student knows sorting and graphs."}],
        "contents": [
            {"id": "c1", "text": "Module 1", "heading": True, "items": [{"id": "c1.1", "text": "Sorting"}, {"id": "c1.2", "text": "Graphs"}]},
            {"id": "c2", "text": "Case study"},
        ],
        "bibliography": "",
    }


def topic(key: str, evidence_ids: list[str], share: float) -> dict:
    return {
        "key": key,
        "name": key.replace("_", " "),
        "description": f"Covers {key}.",
        "evidence": evidence_ids,
        "share": share,
        "level": "introductory",
        "kind": "technique",
        "technology": None,
    }


GOOD = {
    "topics": [topic("sorting_algorithms", ["c1.1"], 0.5), topic("graph_traversal", ["c1.2", "o1"], 0.5)],
    "not_topics": [{"id": "c2", "reason": "no specific content"}],
}


class FakeBackend:
    def __init__(self, url: str, replies: list) -> None:
        self.url = url
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages, request, schema) -> Completion:
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else GOOD
        if isinstance(reply, Exception):
            raise reply
        return Completion(reply if isinstance(reply, str) else json.dumps(reply), {"completion_tokens": 100}, 2.0)


def settings() -> EndpointsFile:
    return EndpointsFile(model="qwen-test", endpoints=[EndpointSpec(url="http://a")], request={"temperature": 0})


class PromptTest(unittest.TestCase):
    def test_messages_list_items_with_ids_and_the_expected_range(self) -> None:
        system, user = course_messages(evidence(credits=6))
        self.assertEqual(system["role"], "system")
        self.assertIn("about 48 hours of lectures), so expect between 6 and 12 topics", user["content"])
        self.assertIn("[c1] Module 1\n  [c1.1] Sorting\n  [c1.2] Graphs\n[c2] Case study", user["content"])
        self.assertIn("Physics (first cycle)", user["content"])
        self.assertIn("Bibliography (context only):\n(none)", user["content"])
        self.assertTrue(render_bibliography("x" * 3000).endswith("[... bibliography truncated]"))

    def test_schema_has_no_references(self) -> None:
        schema = json.dumps(response_schema())
        self.assertNotIn("$ref", schema)
        self.assertNotIn("$defs", schema)
        self.assertIn('"enum": ["introductory", "intermediate", "advanced"]', schema)

    def test_valid_reply(self) -> None:
        self.assertEqual(validate_extraction(Extraction.model_validate(GOOD), evidence()), [])

    def test_heading_citation_covers_its_items(self) -> None:
        reply = {"topics": [topic("sorting_and_graphs", ["c1"], 0.9), topic("case_studies", ["c2"], 0.1)], "not_topics": []}
        self.assertEqual(validate_extraction(Extraction.model_validate(reply), evidence()), [])

    def test_invalid_replies_explain_every_problem(self) -> None:
        reply = {
            "topics": [
                topic("Sorting", ["c1.1", "c9"], 0.3),
                topic("graphs", [], 0.3),
                topic("graphs", ["c2"], 0.0),
            ],
            "not_topics": [{"id": "c2", "reason": "none"}],
        }
        errors = validate_extraction(Extraction.model_validate(reply), evidence())
        expected = [
            "the key graphs is used by several topics",
            "the key 'Sorting' is not lowercase snake_case",
            "the topic graphs cites no syllabus item",
            "the topic graphs has a share of 0.0; shares must be positive",
            "these ids do not exist in the syllabus: c9",
            "these items are both cited by a topic and listed in not_topics: c2",
            "these content items are neither cited by a topic nor listed in not_topics: c1.2",
            "the shares add up to 0.60 instead of 1",
        ]
        self.assertEqual(errors, expected)

    def test_keys_name_the_concept_not_the_technology(self) -> None:
        self.assertTrue(names_technology("python_data_types", "Python"))
        self.assertTrue(names_technology("input_output_in_c", "C"))
        self.assertTrue(names_technology("cpp_templates", "C++"))
        self.assertTrue(names_technology("unix_shell_scripting", "Unix shell"))
        self.assertFalse(names_technology("unix_command_line", "Unix shell"))
        self.assertFalse(names_technology("recursion", "C"))
        reply = json.loads(json.dumps(GOOD))
        reply["topics"][0].update(key="java_sorting", technology="Java")
        reply["topics"][1].update(key="graph_traversal", technology="Java", kind="language")
        errors = validate_extraction(Extraction.model_validate(reply), evidence())
        self.assertEqual(
            errors,
            ["the key java_sorting names its technology Java; keys name the concept and the technology goes only in the technology field"],
        )

    def test_topic_count_follows_the_credits(self) -> None:
        reply = {"topics": [topic("everything", ["c1", "c2"], 1.0)], "not_topics": []}
        errors = validate_extraction(Extraction.model_validate(reply), evidence(credits=6))
        self.assertEqual(errors, ["1 topics for a 6-ECTS course; topics of 4 to 8 hours give between 6 and 12"])


class Handler(http.server.BaseHTTPRequestHandler):
    reply: dict = {}
    status = 200
    bodies: list = []

    def do_GET(self) -> None:
        self._answer({"data": [{"id": "qwen-a"}, {"id": "qwen-b"}]})

    def do_POST(self) -> None:
        Handler.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self._answer(Handler.reply)

    def _answer(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(Handler.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.endpoint = Endpoint(EndpointSpec(url=f"http://127.0.0.1:{cls.server.server_port}"), "qwen", timeout=5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        Handler.status = 200
        Handler.bodies = []

    def test_completion_request_and_reply(self) -> None:
        Handler.reply = {
            "choices": [{"message": {"content": "<think>hmm</think>\n{\"topics\": []}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        completion = self.endpoint.complete([{"role": "user", "content": "hi"}], {"temperature": 0}, {"type": "object"})
        self.assertEqual(completion.content, '{"topics": []}')
        self.assertEqual(completion.usage, {"prompt_tokens": 10, "completion_tokens": 5})
        body = Handler.bodies[0]
        self.assertEqual(body["model"], "qwen")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertEqual(self.endpoint.models(), ["qwen-a", "qwen-b"])

    def test_errors(self) -> None:
        Handler.reply = {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]}
        with self.assertRaisesRegex(RequestError, "max_tokens"):
            self.endpoint.complete([], {}, {})
        Handler.status = 400
        with self.assertRaises(RequestError):
            self.endpoint.complete([], {}, {})
        Handler.status = 503
        with self.assertRaises(EndpointError):
            self.endpoint.complete([], {}, {})
        with self.assertRaises(EndpointError):
            Endpoint(EndpointSpec(url="http://127.0.0.1:9"), "qwen", timeout=2).models()

    def test_helpers(self) -> None:
        self.assertEqual(api_base("http://host:8080/"), "http://host:8080/v1")
        self.assertEqual(api_base("http://host/v1"), "http://host/v1")
        self.assertEqual(strip_thinking("<think>a\nb</think> x"), "x")


class RunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.directory.name)
        self.output = root / "out"
        self.cache = root / "cache"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def extractor(self, **config) -> Extractor:
        return Extractor(settings(), self.output, self.cache, RunConfig(**config))

    def load(self, name: str) -> dict:
        return yaml.safe_load((self.output / name).read_text(encoding="utf-8"))

    def test_retry_with_errors_then_cache_replay(self) -> None:
        bad = {"topics": [topic("sorting_algorithms", ["c1.1"], 1.0)], "not_topics": []}
        backend = FakeBackend("http://a", ["not json", bad, GOOD])
        results = self.extractor().run({"1": evidence()}, ["1"], [(backend, 1)])
        self.assertEqual(results["1"].status, "ok")
        self.assertEqual(results["1"].attempts, 3)
        self.assertIn("not valid JSON", backend.calls[1][-1]["content"])
        self.assertIn("c1.2, c2", backend.calls[2][-1]["content"])
        saved = self.load("topics-1.yml")
        self.assertEqual(saved["format"], "extraction/1")
        self.assertEqual(saved["run"]["endpoint"], "http://a")
        self.assertEqual(saved["run"]["usage"], {"completion_tokens": 300})
        self.assertEqual([entry["key"] for entry in saved["topics"]], ["sorting_algorithms", "graph_traversal"])
        manifest = self.load("run.yml")
        self.assertEqual(manifest["summary"], {"ok": 1})
        self.assertEqual(manifest["servers"]["http://a"]["tokens_per_second"], 50.0)

        replay = FakeBackend("http://a", [])
        self.assertEqual(self.extractor().run({"1": evidence()}, ["1"], [(replay, 1)])["1"].status, "skipped")
        results = self.extractor(force=True).run({"1": evidence()}, ["1"], [(replay, 1)])
        self.assertEqual((results["1"].status, results["1"].endpoint, replay.calls), ("ok", "cache", []))

    def test_failure_is_recorded(self) -> None:
        backend = FakeBackend("http://a", ["{}", "{}"])
        results = self.extractor(max_attempts=2).run({"1": evidence()}, ["1"], [(backend, 1)])
        self.assertEqual(results["1"].status, "failed")
        failure = self.load("failed-1.yml")
        self.assertEqual(failure["last_reply"], "{}")
        self.assertTrue(any("topics" in error for error in failure["errors"]))
        self.assertFalse((self.output / "topics-1.yml").exists())

    def test_unreachable_endpoint_hands_courses_to_the_others(self) -> None:
        dead = FakeBackend("http://dead", [EndpointError("down")])
        alive = FakeBackend("http://alive", [])
        evidences = {course: evidence(course) for course in ("1", "2", "3")}
        results = self.extractor(max_endpoint_failures=1).run(evidences, ["1", "2", "3"], [(dead, 1), (alive, 2)])
        self.assertEqual({result.status for result in results.values()}, {"ok"})
        self.assertEqual({result.endpoint for result in results.values()}, {"http://alive"})

    def test_no_endpoint_left(self) -> None:
        dead = FakeBackend("http://dead", [EndpointError("down")] * 5)
        results = self.extractor(max_endpoint_failures=1).run({"1": evidence()}, ["1"], [(dead, 1)])
        self.assertEqual(results["1"].errors, ["no endpoint left to serve the request"])

    def test_shared_syllabus_is_extracted_once(self) -> None:
        evidences = {"1": evidence("1", same=["2"]), "2": evidence("2", same=["1"])}
        self.assertEqual(syllabus_groups(evidences, ["1", "2"]), {"1": ["2"]})
        backend = FakeBackend("http://a", [])
        results = self.extractor().run(evidences, ["1", "2"], [(backend, 1)])
        self.assertEqual((results["1"].status, results["2"].status, len(backend.calls)), ("ok", "copied", 1))
        self.assertEqual(self.load("topics-2.yml")["copied_from"], "1")

    def test_course_selection(self) -> None:
        evidences = {"1": evidence("1"), "2": evidence("2", service=True), "3": evidence("3", service=True)}
        self.assertEqual(select_courses(evidences, service=True, limit=1), ["2"])
        self.assertEqual(select_courses(evidences, courses=("3", "1")), ["1", "3"])
        with self.assertRaisesRegex(ValueError, "No evidence for courses: 9"):
            select_courses(evidences, courses=("9",))

    def test_cli_and_example_endpoints(self) -> None:
        args = build_parser().parse_args(["run", "ev", "out", "--endpoints", "e.yml", "--course", "1", "--service"])
        self.assertEqual((args.course, args.service, args.max_attempts), (["1"], True, 3))
        example = load_endpoints(pathlib.Path("extraction/endpoints.example.yml"))
        self.assertEqual([spec.workers for spec in example.endpoints], [1, 2])


FAKE_CLAUDE = """#!{python}
import json, os, pathlib, sys
folder = pathlib.Path({folder!r})
replies = json.loads((folder / "replies.json").read_text())
reply = replies.pop(0)
(folder / "replies.json").write_text(json.dumps(replies))
calls = folder / "calls.jsonl"
with calls.open("a") as log:
    log.write(json.dumps({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd(),
                          "no_claude_md": os.environ.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS")}}) + "\\n")
print(reply["stdout"] if isinstance(reply["stdout"], str) else json.dumps(reply["stdout"]))
sys.exit(reply.get("code", 0))
"""


def cli_result(output: dict | None = None, subtype: str = "success", model: str = "claude-haiku-5-5") -> dict:
    result = {
        "type": "result",
        "subtype": subtype,
        "is_error": subtype != "success",
        "result": "",
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 800,
                  "output_tokens": 300, "output_tokens_details": {"thinking_tokens": 120}},
        "modelUsage": {model: {}},
    }
    if output is not None:
        result["structured_output"] = output
    return result


class ClaudeCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.folder = pathlib.Path(self.temp.name)
        self.script = self.folder / "claude"
        self.script.write_text(FAKE_CLAUDE.format(python=sys.executable, folder=str(self.folder)))
        self.script.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def endpoint(self, replies: list[dict], request: dict | None = None) -> ClaudeCliEndpoint:
        (self.folder / "replies.json").write_text(json.dumps(replies))
        spec = EndpointSpec(provider="claude-cli", command=str(self.script))
        return ClaudeCliEndpoint(spec, "claude-haiku-5-5", 30, request or {})

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in (self.folder / "calls.jsonl").read_text().splitlines()]

    def test_isolated_request_and_structured_reply(self) -> None:
        endpoint = self.endpoint([{"stdout": cli_result(GOOD)}], {"effort": "low"})
        completion = endpoint.complete(course_messages(evidence()), {"effort": "low"}, response_schema())
        self.assertEqual(json.loads(completion.content), GOOD)
        self.assertEqual(
            completion.usage,
            {"prompt_tokens": 1502, "completion_tokens": 300, "cache_read_tokens": 800, "thinking_tokens": 120},
        )
        call = self.calls()[0]
        argv = call["argv"]
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project")
        self.assertEqual(argv[argv.index("--effort") + 1], "low")
        self.assertIn("--strict-mcp-config", argv)
        self.assertEqual(json.loads(argv[argv.index("--json-schema") + 1]), response_schema())
        self.assertIn("[c1.1] Sorting", call["stdin"])
        self.assertEqual(call["no_claude_md"], "1")
        self.assertEqual(os.listdir(call["cwd"]), [])

    def test_retries_carry_the_conversation(self) -> None:
        messages = course_messages(evidence()) + [{"role": "assistant", "content": '{"topics": []}'}, retry_message(["too few topics"])]
        system, prompt = flatten(messages)
        self.assertEqual(system, messages[0]["content"])
        self.assertTrue(prompt.startswith(messages[1]["content"]))
        self.assertIn('Your reply number 1 was:\n\n{"topics": []}', prompt)
        self.assertIn("- too few topics", prompt)

    def test_errors(self) -> None:
        endpoint = self.endpoint([
            {"stdout": cli_result(subtype="error_max_structured_output_retries")},
            {"stdout": cli_result(subtype="error_during_execution")},
            {"stdout": "not json", "code": 1},
            {"stdout": cli_result(GOOD, model="claude-sonnet-5-5")},
        ])
        messages = course_messages(evidence())
        with self.assertRaises(RequestError):
            endpoint.complete(messages, {}, response_schema())
        with self.assertRaises(EndpointError):
            endpoint.complete(messages, {}, response_schema())
        with self.assertRaises(EndpointError):
            endpoint.complete(messages, {}, response_schema())
        with self.assertRaises(RequestError):
            endpoint.complete(messages, {}, response_schema())

    def test_runs_through_the_extractor(self) -> None:
        settings = EndpointsFile(model="claude-haiku-5-5", endpoints=[EndpointSpec(provider="claude-cli", command=str(self.script))])
        (self.folder / "replies.json").write_text(json.dumps([{"stdout": cli_result({"topics": [], "not_topics": []})}, {"stdout": cli_result(GOOD)}]))
        [endpoint] = make_endpoints(settings)
        output = self.folder / "out"
        results = Extractor(settings, output, self.folder / "cache").run({"1": evidence()}, ["1"], [(endpoint, 1)])
        self.assertEqual((results["1"].status, results["1"].attempts, results["1"].endpoint), ("ok", 2, "claude-cli"))
        self.assertIn("Your reply number 1 was", self.calls()[1]["stdin"])

    def test_configuration_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.endpoint([], {"temperature": 0})
        with self.assertRaises(ValueError):
            EndpointSpec(workers=2)


class ManualTest(unittest.TestCase):
    def test_requests_and_ingest(self) -> None:
        evidences = {"1": evidence("1", same=["2"]), "2": evidence("2", same=["1"])}
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            written = write_requests(evidences, ["1", "2"], output)
            self.assertEqual([path.name for path in written], ["request-1.md"])
            request = written[0].read_text(encoding="utf-8")
            self.assertIn("[c1.1] Sorting", request)
            self.assertIn('"not_topics"', request)
            reply = output / "reply.json"
            reply.write_text("```json\n" + json.dumps({"topics": GOOD["topics"], "not_topics": []}) + "\n```", encoding="utf-8")
            errors = ingest_reply(evidences, output, "1", reply, "claude-test")
            self.assertEqual(errors, ["these content items are neither cited by a topic nor listed in not_topics: c2"])
            reply.write_text(json.dumps(GOOD), encoding="utf-8")
            self.assertEqual(ingest_reply(evidences, output, "1", reply, "claude-test"), [])
            saved = yaml.safe_load((output / "topics-1.yml").read_text(encoding="utf-8"))
            self.assertEqual((saved["run"]["model"], saved["run"]["attempts"]), ("claude-test", 2))
            self.assertEqual(yaml.safe_load((output / "topics-2.yml").read_text(encoding="utf-8"))["copied_from"], "1")
            manifest = yaml.safe_load((output / "run.yml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"], {"ok": 1, "copied": 1})
            self.assertFalse((output / ".attempts" / "1.json").exists())


if __name__ == "__main__":
    unittest.main()
