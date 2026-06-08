from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import subprocess
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def generate_experiment_id(prefix: str = "exp") -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def collect_git_info(repo_root: pathlib.Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip() or None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "--short", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status,
    }


def write_json(path: pathlib.Path, data: Any) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: pathlib.Path, record: Any) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
    return path


def append_markdown_experiment(
    path: pathlib.Path,
    *,
    experiment_id: str,
    started_at: str,
    command: str,
    settings: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    conclusion: str = "TODO",
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"## {experiment_id}",
        f"Date: {started_at}",
        "",
        "What changed:",
        f"- {settings.get('summary', 'See manifest.json for the full config.')}",
        "",
        "Why:",
        f"- {settings.get('why', 'Recorded automatically from the training command.')}",
        "",
        "Training settings:",
        "```text",
        command,
        "```",
        "",
        "Results:",
    ]
    if metrics:
        for key, value in metrics.items():
            sections.append(f"- {key}: {value}")
    else:
        sections.append("- See `summary.json`.")
    sections.extend([
        "",
        "Conclusion:",
        f"- {conclusion}",
        "",
    ])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(sections))
    return path

