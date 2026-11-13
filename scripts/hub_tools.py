#!/usr/bin/env python3
"""Checks and publishing of the model hub. Standard library only: no weights are ever loaded,
the safetensors header is read as data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SPORT_CLASSES = {
    "football": {"ball", "player", "goalkeeper", "referee", "goal"},
    "futsal": {"ball", "player", "goalkeeper", "referee", "goal"},
    "basketball": {"ball", "player", "referee", "hoop"},
    "handball": {"ball", "player", "goalkeeper", "referee", "goal"},
    "hockey": {"puck", "player", "goalkeeper", "referee", "goal"},
    "rugby": {"ball", "player", "referee", "goalpost"},
    "lacrosse": {"ball", "player", "goalkeeper", "referee", "goal"},
    "american_football": {"ball", "player", "referee", "goalpost"},
}
SIZES = {"nano": 384, "small": 512, "medium": 576}
# YOLO26 (Ultralytics, AGPL-3.0). The size names carry the family, so one folder level holds both.
YOLO_SIZES = {"yolo-nano": 16, "yolo-small": 32, "yolo-medium": 64}  # width of the first convolution
YOLO_FIRST_KEY = "model.0.conv.weight"
YOLO_CLASS_KEYS = tuple(f"model.23.cv3.{head}.2.weight" for head in range(3))
YOLO_IMAGE_SIZES = (640, 960, 1280)
YOLO_TENSOR_KEY = re.compile(r"^model\.\d+\.[A-Za-z0-9_.]+$")
YOLO_DTYPES = {"F16", "F32", "I64"}
YOLO_TRAINER_MIN = (0, 37, 0)
# Which license, which rights statement and which sizes belong to which architecture.
ARCHITECTURES = {
    "rf-detr": {"license": "Apache-2.0", "statement": 1, "sizes": tuple(SIZES)},
    "yolo26": {"license": "AGPL-3.0", "statement": 2, "sizes": tuple(YOLO_SIZES)},
}
POSITION_TOKENS = {577: "nano", 1025: "small", 1297: "medium"}
POSITION_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"
CLASS_HEAD_KEY = "transformer.enc_out_class_embed.0.bias"
MAX_WEIGHTS_BYTES = 500_000_000
RIGHTS_STATEMENT_VERSION = 1
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TENSOR_KEY = re.compile(r"^[A-Za-z0-9_.]+$")
PRIVATE_TEXT = re.compile(r"(/Users/|/home/|[A-Za-z]:\\|/Volumes/|\S+@\S+\.\S+)")
MODEL_PATH = re.compile(r"^sports/([a-z_]+)/([a-z]+(?:-[a-z]+)?)/([a-z0-9-]+)/model\.json$")
ALLOWED_KEYS = {"schema", "id", "name", "version", "sport", "size", "architecture", "parent", "author", "license", "description", "classes", "input_resolution", "training", "checkpoint_sanitized", "rights", "metrics", "compat", "files", "deprecated", "framework"}
INDEX_DROPPED = {"rights"}


class Invalid(Exception):
    pass


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(version).split("."))


def check_changed_files(changed: list[str]) -> str:
    """A pull request changes exactly one model.json and nothing else. Returns its path."""
    if len(changed) != 1 or not MODEL_PATH.match(changed[0]):
        raise Invalid(f"a pull request may only change one sports/<sport>/<size>/<model>/model.json, not: {', '.join(changed[:10]) or 'nothing'}")
    return changed[0]


def check_model(model: dict, path: str, author: str | None = None) -> None:
    problems = []

    def need(condition, message: str) -> None:
        if not condition:
            problems.append(message)

    if not isinstance(model, dict):
        raise Invalid("model.json is not an object")
    unknown = set(model) - ALLOWED_KEYS
    need(not unknown, f"unknown fields: {sorted(unknown)}")
    sport, size, model_id = MODEL_PATH.match(path).groups()
    need(model.get("schema") == 1, "schema must be 1")
    need(model.get("sport") == sport and sport in SPORT_CLASSES, f"sport must match the folder and be one of {sorted(SPORT_CLASSES)}")
    architecture = ARCHITECTURES.get(model.get("architecture"))
    need(architecture is not None, "architecture must be rf-detr or yolo26")
    architecture = architecture or ARCHITECTURES["rf-detr"]
    need(model.get("size") == size and size in architecture["sizes"], f"size must match the folder and be one of {', '.join(architecture['sizes'])}")
    need(model.get("id") == model_id and bool(MODEL_ID.match(model_id)), "id must match the folder")
    need(bool(VERSION.match(str(model.get("version", "")))), "version must look like 1.2.3")
    need(model.get("license") == architecture["license"], f"license must be {architecture['license']}")
    need(model.get("checkpoint_sanitized") is True, "checkpoint_sanitized must be true")
    need(isinstance(model.get("author"), str) and bool(model.get("author")), "author is missing")
    if author is not None:
        need(model.get("author") == author, "author must be the GitHub account of the pull request")
    if model.get("architecture") == "yolo26":
        need(model.get("input_resolution") in YOLO_IMAGE_SIZES, f"input_resolution must be one of {YOLO_IMAGE_SIZES}")
        framework = model.get("framework")
        need(isinstance(framework, dict) and framework.get("name") == "ultralytics" and bool(re.match(r"^\d+\.\d+\.\d+$", str(framework.get("version", "")))), "framework must name ultralytics and its version")
    else:
        need(model.get("input_resolution") == SIZES.get(size), "input_resolution does not fit the size")
        need("framework" not in model, "framework is only for yolo26 models")
    classes = model.get("classes")
    need(isinstance(classes, list) and bool(classes) and len(set(classes)) == len(classes) and set(classes) <= SPORT_CLASSES.get(sport, set()), "classes must be classes of this sport")
    for field in ("name", "description"):
        need(isinstance(model.get(field), str) and len(model.get(field, "")) <= 400, f"{field} must be text")
        need(not PRIVATE_TEXT.search(str(model.get(field, ""))), f"{field} must not contain paths or mail addresses")
    need(bool(model.get("name")), "name is missing")
    parent = model.get("parent")
    need(parent is None or re.match(r"^[a-z_]+/(nano|small|medium|yolo-nano|yolo-small|yolo-medium)/[a-z0-9-]+@\d+\.\d+\.\d+$", str(parent)), "parent must look like sport/size/model@1.2.3")
    if parent and re.match(r"^[a-z_]+/([a-z-]+)/", str(parent)):
        need((re.match(r"^[a-z_]+/([a-z-]+)/", str(parent)).group(1) in YOLO_SIZES) == (model.get("architecture") == "yolo26"), "parent must be a model of the same architecture")
    training = model.get("training")
    need(isinstance(training, dict) and isinstance(training.get("images"), int) and isinstance(training.get("games"), int) and training.get("games", 0) >= 1 and bool(training.get("camera")) and training.get("venue") in {"indoor", "outdoor"}, "training needs images, games, camera and venue")
    need(not PRIVATE_TEXT.search(json.dumps(training or {})), "training must not contain paths or mail addresses")
    rights = model.get("rights")
    need(isinstance(rights, dict) and rights.get("confirmed_by") == model.get("author") and bool(rights.get("confirmed_at")) and rights.get("statement_version") == architecture["statement"], "the confirmation of the rights is missing")
    trainer_min = str((model.get("compat") or {}).get("trainer_min", ""))
    need(bool(re.match(r"^\d+\.\d+\.\d+$", trainer_min)), "compat.trainer_min is missing")
    if model.get("architecture") == "yolo26" and re.match(r"^\d+\.\d+\.\d+$", trainer_min):
        need(version_key(trainer_min) >= YOLO_TRAINER_MIN, "compat.trainer_min must be at least 0.37.0 for yolo26 models")
    files = model.get("files")
    if not (isinstance(files, list) and len(files) == 1 and isinstance(files[0], dict) and files[0].get("format") == "safetensors"):
        problems.append("files must hold exactly one safetensors file")
    else:
        need(bool(SHA256.match(str(files[0].get("sha256", "")))), "sha256 is missing")
        tag = f"{sport}-{size}-{model_id}-v{model.get('version')}"
        need(str(files[0].get("source_url", "")).startswith(f"https://github.com/{model.get('author')}/reco-models/releases/download/{tag}/"), "source_url must be a release of the author's fork with the right tag")
    if problems:
        raise Invalid("; ".join(problems))


def check_against_base(model: dict, base_file: Path) -> None:
    """Only the author updates a model, and only to a newer version."""
    if not base_file.is_file():
        return
    existing = json.loads(base_file.read_text(encoding="utf-8"))
    if existing.get("author") != model.get("author"):
        raise Invalid(f"this model belongs to {existing.get('author')}; choose another name")
    if version_key(model["version"]) <= version_key(existing["version"]):
        raise Invalid(f"the version must be newer than {existing['version']}")


def read_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Invalid("the weights are not a safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if length > min(size - 8, 10_000_000):
            raise Invalid("the weights are not a safetensors file")
        try:
            header = json.loads(handle.read(length))
        except ValueError as error:
            raise Invalid("the weights are not a safetensors file") from error
    if header.pop("__metadata__", {}) not in ({}, {"format": "pt"}):
        raise Invalid("the weights carry unexpected metadata")
    end = 0
    for name, tensor in header.items():
        if not TENSOR_KEY.match(name):
            raise Invalid(f"unexpected tensor name: {name[:80]}")
        end = max(end, int(tensor["data_offsets"][1]))
    if 8 + length + end != size:
        raise Invalid("the weights file holds more than its tensors")
    return header


def check_weights(path: Path, model: dict) -> None:
    entry = model["files"][0]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise Invalid("the checksum of the weights does not match")
    header = read_header(path)
    if model.get("architecture") == "yolo26":
        check_yolo_header(header, model)
        return
    try:
        tokens, head = header[POSITION_KEY]["shape"][1], header[CLASS_HEAD_KEY]["shape"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise Invalid("these weights are not an RF-DETR detection model") from error
    if POSITION_TOKENS.get(tokens) != model["size"]:
        raise Invalid(f"the weights are {POSITION_TOKENS.get(tokens, 'not Nano, Small or Medium')}, not {model['size']}")
    if head - 1 != len(model["classes"]):
        raise Invalid(f"the weights have {head - 1} classes, model.json lists {len(model['classes'])}")


def check_yolo_header(header: dict, model: dict) -> None:
    """The weights of a YOLO26 detector, judged from the header alone: the width of the first
    convolution gives the size, the three class heads give the number of classes."""
    for name, tensor in header.items():
        if not YOLO_TENSOR_KEY.match(name):
            raise Invalid(f"unexpected tensor name: {name[:80]}")
        if tensor.get("dtype") not in YOLO_DTYPES:
            raise Invalid("the weights hold tensors of an unexpected type")
    if not 500 <= len(header) <= 1000:
        raise Invalid("these weights are not a YOLO26 detection model")
    try:
        width = header[YOLO_FIRST_KEY]["shape"][0]
        heads = [header[key]["shape"] for key in YOLO_CLASS_KEYS]
    except (KeyError, IndexError, TypeError) as error:
        raise Invalid("these weights are not a YOLO26 detection model") from error
    size = {value: name for name, value in YOLO_SIZES.items()}.get(width)
    if size is None:
        raise Invalid("only YOLO26 Nano, Small and Medium can be shared")
    if size != model["size"]:
        raise Invalid(f"the weights are {size}, not {model['size']}")
    counts = {shape[0] for shape in heads}
    if len(counts) != 1 or any(len(shape) != 4 or shape[1] != 4 * width for shape in heads):
        raise Invalid("these weights are not a YOLO26 detection model")
    if counts.pop() != len(model["classes"]):
        raise Invalid(f"the weights have a different number of classes than model.json lists ({len(model['classes'])})")


def download(url: str, destination: Path) -> None:
    written = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "reco-models"}), timeout=300) as response, destination.open("wb") as sink:
        while chunk := response.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_WEIGHTS_BYTES:
                raise Invalid("the weights are too large")
            sink.write(chunk)


def validate(pr: Path, base: Path, changed: list[str], author: str | None) -> dict:
    path = check_changed_files(changed)
    try:
        model = json.loads((pr / path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Invalid(f"{path} cannot be read: {error}") from error
    check_model(model, path, author)
    check_against_base(model, base / path)
    with tempfile.TemporaryDirectory() as folder:
        weights = Path(folder) / "weights.safetensors"
        download(model["files"][0]["source_url"], weights)
        check_weights(weights, model)
    return model


def all_models(root: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "sports").glob("*/*/*/model.json"))]


def tag_of(model: dict) -> str:
    return f"{model['sport']}-{model['size']}-{model['id']}-v{model['version']}"


def build_index(models: list[dict], repo: str) -> dict:
    """Rights stay in the repository; the trainer only sees where to get the weights from."""
    entries = []
    for model in models:
        entry = {key: value for key, value in model.items() if key not in INDEX_DROPPED}
        file = {key: value for key, value in model["files"][0].items() if key != "source_url"}
        name = str(model["files"][0]["source_url"]).rsplit("/", 1)[-1]
        file["url"] = f"https://github.com/{repo}/releases/download/{tag_of(model)}/{name}"
        entry["files"] = [file]
        entries.append(entry)
    return {"schema": 1, "repo": repo, "models": entries}


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, capture_output=True, text=True)


def publish(root: Path, repo: str) -> None:
    models = all_models(root)
    with tempfile.TemporaryDirectory() as folder:
        for model in models:
            tag = tag_of(model)
            if gh("release", "view", tag, "--repo", repo, check=False).returncode == 0:
                continue
            source = model["files"][0]["source_url"]
            weights = Path(folder) / source.rsplit("/", 1)[-1]
            download(source, weights)
            # Assets of a fork can be replaced: the hash from the reviewed model.json decides.
            check_weights(weights, model)
            gh("release", "create", tag, str(weights), "--repo", repo, "--title", f"{model['name']} {model['version']}", "--notes", f"{model['sport']}/{model['size']}/{model['id']} by {model['author']}. See LICENSE and NOTICE.", "--latest=false")
            weights.unlink()
        index = Path(folder) / "index.json"
        index.write_text(json.dumps(build_index(models, repo), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if gh("release", "view", "index", "--repo", repo, check=False).returncode != 0:
            gh("release", "create", "index", "--repo", repo, "--title", "Index", "--notes", "Generated. Read by Reco Trainer.", "--latest=false")
        gh("release", "upload", "index", str(index), "--repo", repo, "--clobber")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--pr", type=Path, required=True)
    check.add_argument("--base", type=Path, required=True)
    check.add_argument("--changed", type=Path, required=True)
    check.add_argument("--author")
    release = commands.add_parser("publish")
    release.add_argument("--repo", required=True)
    release.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        if args.command == "validate":
            changed = [line for line in args.changed.read_text(encoding="utf-8").splitlines() if line]
            model = validate(args.pr, args.base, changed, args.author)
            print(f"OK: {model['sport']}/{model['size']}/{model['id']} {model['version']}")
        else:
            publish(args.root, args.repo)
    except Invalid as error:
        print(f"::error::{error}")
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
