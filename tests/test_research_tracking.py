import argparse
import json
import pathlib

import scripts.train as train_script
from research.tracking import append_jsonl, collect_git_info, generate_experiment_id, write_json


def test_generate_experiment_id_prefix():
    experiment_id = generate_experiment_id("E")

    assert experiment_id.startswith("E-")
    assert len(experiment_id) > 10


def test_write_json_and_append_jsonl(tmp_path):
    json_path = tmp_path / "manifest.json"
    jsonl_path = tmp_path / "metrics.jsonl"

    write_json(json_path, {"path": pathlib.Path("x"), "value": 1})
    append_jsonl(jsonl_path, {"step": 1, "loss": 2.5})
    append_jsonl(jsonl_path, {"step": 2, "loss": 2.0})

    data = json.loads(json_path.read_text())
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

    assert data["path"] == "x"
    assert lines[0]["step"] == 1
    assert lines[1]["loss"] == 2.0


def test_initialize_research_record_writes_manifest(tmp_path, monkeypatch):
    args = argparse.Namespace(
        research_root=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        buffer_path=tmp_path / "buffer.npz",
        device="cpu",
    )
    monkeypatch.setattr(train_script, "CKPT_DIR", args.checkpoint_dir)
    monkeypatch.setattr(train_script, "BEST_CKPT_PATH", args.checkpoint_dir / "best_model.pt")

    manifest_path, metrics_path, summary_path = train_script.initialize_research_record(
        args,
        "E-test",
        "python scripts/train.py --experiment_id E-test",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["experiment_id"] == "E-test"
    assert metrics_path.name == "metrics.jsonl"
    assert summary_path.name == "summary.json"
    assert "git" in manifest


def test_collect_git_info_returns_mapping():
    info = collect_git_info(pathlib.Path.cwd())

    assert isinstance(info, dict)
    assert {"commit", "branch", "dirty", "status"} <= set(info)
