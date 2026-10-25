from __future__ import annotations

import re
from pathlib import Path

from cresta.paths import DATA_WORKSPACE

ROOT = Path(__file__).resolve().parents[2]


def test_category_boundaries_exist() -> None:
    for path in (
        "src/tau2",
        "src/cresta",
        "experiments/baseline",
        "experiments/persona_search",
        "benchmarks",
        "datasets/tau2",
        "datasets/usi",
        "services",
        "vendor",
    ):
        assert (ROOT / path).exists(), path


def test_root_has_no_compatibility_symlinks() -> None:
    assert [path.name for path in ROOT.iterdir() if path.is_symlink()] == []


def test_active_code_has_no_old_checkout_paths_or_plaintext_keys() -> None:
    old_checkout = "/Users/yutao.xie/work/tau2bench_cc"
    secret = re.compile(
        r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|"
        r"AIza[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|"
        r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}:[A-Za-z0-9_-]{20,})"
    )
    offenders = []
    for area in ("src", "experiments", "benchmarks", "services"):
        for path in (ROOT / area).rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".toml"}:
                continue
            text = path.read_text(errors="ignore")
            if old_checkout in text or secret.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_non_usi_naturalness_code_is_not_active() -> None:
    active = ROOT / "src" / "cresta"
    banned = re.compile(
        r"\b(?:cc_user|cc_user_con|naturalness_opt|sim2real_llm_metrics|"
        r"persona_behavior_jsd|persona_following_compliance|P_geom)\b"
    )
    offenders = []
    for path in active.rglob("*.py"):
        if banned.search(path.read_text(errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    assert (active / "evaluation" / "usi").is_dir()


def test_vendored_projects_have_no_nested_git_metadata() -> None:
    for path in (
        "vendor/vitabench",
        "vendor/persona_policies",
        "vendor/user_behavioral_divergence",
    ):
        assert not (ROOT / path / ".git").exists(), path


def test_external_workspace_is_outside_git_checkout() -> None:
    assert DATA_WORKSPACE != ROOT
    assert not DATA_WORKSPACE.is_relative_to(ROOT)
