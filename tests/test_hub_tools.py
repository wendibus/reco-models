import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import hub_tools


def weights_file(path: Path, tokens=1025, head=5, extra=b"", key="x", metadata=None) -> None:
    header = {
        hub_tools.POSITION_KEY: {"dtype": "F32", "shape": [1, tokens, 1], "data_offsets": [0, 4]},
        hub_tools.CLASS_HEAD_KEY: {"dtype": "F32", "shape": [head], "data_offsets": [4, 8]},
        key: {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 12 + extra)


def model(**changes) -> dict:
    base = {
        "schema": 1, "id": "hall-model", "name": "Hall model", "version": "1.0.0", "sport": "basketball", "size": "small", "architecture": "rf-detr", "parent": None, "author": "someone",
        "license": "Apache-2.0", "description": "Indoor", "classes": ["ball", "player", "referee", "hoop"], "input_resolution": 512,
        "training": {"images": 10, "games": 2, "camera": "fixed", "venue": "indoor", "epochs": 5}, "checkpoint_sanitized": True,
        "rights": {"confirmed_by": "someone", "confirmed_at": "2026-10-31T00:00:00Z", "statement_version": 1}, "metrics": {"self_reported": {"mAP50": 0.8}}, "compat": {"trainer_min": "0.25.0"},
        "files": [{"format": "safetensors", "sha256": "a" * 64, "size_bytes": 1, "source_url": "https://github.com/someone/reco-models/releases/download/basketball-small-hall-model-v1.0.0/w.safetensors"}],
    }
    return {**base, **changes}


PATH = "sports/basketball/small/hall-model/model.json"


class ChangedFilesTests(unittest.TestCase):
    def test_one_model_json_passes(self):
        self.assertEqual(hub_tools.check_changed_files([PATH]), PATH)

    def test_everything_else_fails(self):
        for changed in (
            [], [PATH, "README.md"], [".github/workflows/validate.yml"], ["scripts/hub_tools.py"], ["sports/basketball/small/hall-model/samples/01.jpg"],
            ["index.json"], [PATH, "sports/basketball/small/other/model.json"], ["sports/basketball/small/hall-model/../../../../x/model.json"],
        ):
            with self.assertRaises(hub_tools.Invalid, msg=changed):
                hub_tools.check_changed_files(changed)


class ModelTests(unittest.TestCase):
    def test_a_good_model_passes(self):
        hub_tools.check_model(model(), PATH, "someone")

    def test_broken_models_fail(self):
        for changes in (
            {"sport": "football"}, {"size": "large"}, {"id": "other"}, {"author": "stranger"}, {"license": "MIT"}, {"classes": ["ball", "goal"]}, {"classes": []}, {"checkpoint_sanitized": False},
            {"description": "from /Users/me/club"}, {"name": "a@b.de"}, {"rights": None}, {"rights": {"confirmed_by": "other", "confirmed_at": "x", "statement_version": 1}}, {"samples": []},
            {"input_resolution": 384}, {"version": "one"}, {"training": {"images": 1}}, {"training": {"images": 1, "games": 1, "camera": "/home/me", "venue": "indoor"}}, {"parent": "../x"},
            {"files": [{"format": "pth", "sha256": "a" * 64, "source_url": "x"}]}, {"files": []}, {"compat": {}},
        ):
            with self.assertRaises(hub_tools.Invalid, msg=changes):
                hub_tools.check_model(model(**changes), PATH, "someone")

    def test_the_url_has_to_be_the_authors_fork(self):
        files = [{"format": "safetensors", "sha256": "a" * 64, "source_url": "https://evil.example/w.safetensors"}]
        with self.assertRaises(hub_tools.Invalid):
            hub_tools.check_model(model(files=files), PATH, "someone")


class BaseTests(unittest.TestCase):
    def check(self, new, existing):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text(json.dumps(existing))
            hub_tools.check_against_base(new, path)

    def test_only_the_author_updates(self):
        with self.assertRaises(hub_tools.Invalid):
            self.check(model(version="2.0.0"), model(author="other"))

    def test_only_newer_versions(self):
        self.check(model(version="1.10.0"), model(version="1.9.0"))
        with self.assertRaises(hub_tools.Invalid):
            self.check(model(version="1.0.0"), model())


class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "w.safetensors"

    def check(self, **file_options):
        weights_file(self.path, **file_options)
        digest = hub_tools.hashlib.sha256(self.path.read_bytes()).hexdigest()
        files = [{**model()["files"][0], "sha256": digest}]
        hub_tools.check_weights(self.path, model(files=files))

    def test_good_weights_pass(self):
        self.check()

    def test_wrong_size_or_classes_fail(self):
        with self.assertRaises(hub_tools.Invalid):
            self.check(tokens=577)
        with self.assertRaises(hub_tools.Invalid):
            self.check(tokens=1937)
        with self.assertRaises(hub_tools.Invalid):
            self.check(head=91)

    def test_hidden_data_fails(self):
        with self.assertRaises(hub_tools.Invalid):
            self.check(extra=b"trailing")
        with self.assertRaises(hub_tools.Invalid):
            self.check(key="/Users/me/path")
        with self.assertRaises(hub_tools.Invalid):
            self.check(metadata={"dataset_dir": "/Users/me"})

    def test_wrong_hash_and_junk_fail(self):
        weights_file(self.path)
        with self.assertRaises(hub_tools.Invalid):
            hub_tools.check_weights(self.path, model())
        self.path.write_bytes(b"\x80\x04 this is a pickle")
        with self.assertRaises(hub_tools.Invalid):
            hub_tools.read_header(self.path)


class IndexTests(unittest.TestCase):
    def test_index_points_to_the_main_repository_and_leaves_rights_out(self):
        index = hub_tools.build_index([model()], "owner/reco-models")
        entry = index["models"][0]
        self.assertNotIn("rights", entry)
        self.assertEqual(entry["files"][0]["url"], "https://github.com/owner/reco-models/releases/download/basketball-small-hall-model-v1.0.0/w.safetensors")
        self.assertNotIn("source_url", entry["files"][0])


if __name__ == "__main__":
    unittest.main()
