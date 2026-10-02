"""The AMD driver's planning logic. No network, no droplet, no GPU, no clock.

Everything worth being wrong about here is a number or an order: what the plan costs, what order
it runs the arms in, and what it refuses to do when the budget is exceeded. Those are all pure
functions, so they are tested directly rather than through a live ssh.

The one thing these tests deliberately do not do is touch the network, which is why `plan()` and
`api_argv()` return argv lists instead of running anything: `--dry-run` prints them, and this file
asserts on them.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.amd.driver import (ARMS, DEFAULT_PRICE_PER_GPU_HOUR, EVAL_HOURS_ESTIMATE,
                            SFT_HOURS_ESTIMATE, Config, api_argv, budget_report, plan,
                            ssh_argv)

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "ops" / "amd" / "driver.py"


def config(**over) -> Config:
    cfg = Config(
        host="203.0.113.9", user="root", identity="/home/me/.ssh/id_ed25519",
        commit="deadbeef", arms=list(ARMS), price=DEFAULT_PRICE_PER_GPU_HOUR, budget=80.0,
        max_minutes=1853, idle_minutes=45, max_length=8192, limit=25, split="test",
        tag_prefix="amd1", remote_root="/opt/smol-ladder", eval_hours=EVAL_HOURS_ESTIMATE,
        smoke_hours=0.5, smoke_limit=5, resume=True, eval_base=True, hub_namespace="me")
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def names(p) -> list[str]:
    return [s.name for s in p.steps]


# ── the budget ────────────────────────────────────────────────────────────────

def test_the_published_price_is_the_official_one_not_the_blogs():
    # $1.99 is what the plan and a third-party blog say; $2.59 is what DigitalOcean's AMD pricing
    # page says. The driver must use the official number, because the difference is 23% of the
    # credit and an underestimate is what runs the window out mid-arm.
    assert DEFAULT_PRICE_PER_GPU_HOUR == 2.59


def test_the_credit_buys_fewer_hours_than_the_plan_assumed():
    hours_at_199 = 100 / 1.99
    hours_at_259 = 100 / DEFAULT_PRICE_PER_GPU_HOUR
    assert hours_at_259 < hours_at_199
    assert round(hours_at_259) == 39  # ~38.6, not ~50


def test_every_budget_row_is_the_price_times_its_hours():
    report = budget_report(2.59, 80.0, list(ARMS), SFT_HOURS_ESTIMATE, 0.75)
    for row in report["rows"]:
        assert row["dollars"] == pytest.approx(row["hours"] * 2.59, abs=0.01), row


def test_the_budget_totals_are_the_sum_of_the_rows():
    report = budget_report(2.59, 80.0, list(ARMS), SFT_HOURS_ESTIMATE, 0.75)
    assert report["total_hours"] == pytest.approx(sum(r["hours"] for r in report["rows"]))
    assert report["total_dollars"] == pytest.approx(sum(r["dollars"] for r in report["rows"]))


def test_the_base_control_is_costed_once_and_not_twice():
    # eval-base appears in the report for the base arm only. It was in the arms list in an earlier
    # draft and double-counted, which is exactly the kind of error that quietly inflates a budget.
    report = budget_report(2.59, 80.0, ["A", "B", "AB"], SFT_HOURS_ESTIMATE, 0.75)
    stages = [r["stage"] for r in report["rows"]]
    assert stages.count("eval-base") == 1
    assert stages.count("eval-A") == 1


def test_dropping_the_base_control_removes_its_row():
    report = budget_report(2.59, 80.0, ["A"], SFT_HOURS_ESTIMATE, 0.75, eval_base=False)
    assert "eval-base" not in [r["stage"] for r in report["rows"]]


def test_an_over_budget_plan_says_so_rather_than_being_reported_as_fine():
    # A 2B at 8192 tokens with an unmeasured throughput is exactly the situation where the
    # estimate is wrong, and the guard has to be on the plan rather than on the owner's memory.
    report = budget_report(2.59, 5.0, list(ARMS), SFT_HOURS_ESTIMATE, 0.75)
    assert not report["within_budget"]
    assert report["headroom_dollars"] < 0


def test_the_full_plan_fits_the_credit_with_room_to_spare():
    report = budget_report(2.59, 80.0, list(ARMS), SFT_HOURS_ESTIMATE, 0.75)
    assert report["within_budget"]
    assert report["headroom_dollars"] > 20  # enough for a second seed or a k=2 sweep


# ── the plan's order and contents ─────────────────────────────────────────────

def test_the_watchdog_is_started_before_anything_that_can_hang():
    order = names(plan(config()))
    assert order.index("watchdog") < order.index("sft-A")
    assert order.index("watchdog") < order.index("smoke-sft")


def test_the_smoke_test_precedes_every_arm():
    order = names(plan(config()))
    assert order.index("smoke-sft") < order.index("sft-A")
    assert order.index("smoke-eval") < order.index("sft-A")


def test_each_arm_is_trained_evaluated_and_synced_before_the_next_arm_starts():
    order = names(plan(config()))
    assert order.index("sft-A") < order.index("eval-A") < order.index("results-A")
    assert order.index("results-A") < order.index("sft-B")
    assert order.index("sft-B") < order.index("eval-B") < order.index("results-B")
    assert order.index("results-B") < order.index("sft-AB")


def test_teardown_is_last_and_powers_off_before_nothing_can_run_after_it():
    order = names(plan(config()))
    assert order[-1] == "poweroff"
    assert order.index("sync-final") < order.index("poweroff")


def test_the_base_control_is_evaluated_and_nothing_more_is_trained_for_it():
    p = plan(config())
    assert "eval-base" in names(p)
    assert not any(n.startswith("sft-") and "base" in n for n in names(p))


def test_the_plan_ends_by_powering_off_and_says_that_billing_continues():
    # The single most expensive mistake available here is treating poweroff as teardown.
    power = plan(config()).steps[-1]
    assert power.argv[-2:] == ["-h", "now"]
    assert "DESTROYED" in power.note


def test_an_arm_is_resumed_not_restarted_by_default():
    p = plan(config())
    assert "--resume" in p.steps[names(p).index("sft-A")].argv


def test_each_arms_adapter_is_pushed_before_the_next_arm_is_trained():
    order = names(plan(config()))
    # A killed run's only surviving copy is on the Hub, so the push cannot come after the next arm.
    assert order.index("sync-A") < order.index("sft-B")
    assert order.index("sync-B") < order.index("sft-AB")


def test_an_over_budget_plan_carries_a_warning():
    p = plan(config(budget=5.0))
    assert any("OVER BUDGET" in w for w in p.warnings)


def test_a_plan_without_a_hub_namespace_says_the_adapter_is_ephemeral():
    p = plan(config(hub_namespace=""))
    assert any("only copy" in w for w in p.warnings)


def test_the_commit_is_pinned_into_the_bootstrap_command():
    p = plan(config(commit="cafe1234"))
    assert "cafe1234" in p.steps[0].argv


def test_no_step_carries_a_secret_in_its_argv():
    # The token lives in the remote .env and is read there. A secret in argv is a secret in the
    # shell history, in `ps` on the laptop, and in this test's failure output.
    for step in plan(config()).steps:
        joined = " ".join(step.argv)
        for name in ("HF_TOKEN", "AMD_CLOUD_API_TOKEN", "DIGITALOCEAN_TOKEN"):
            assert name not in joined, step.name


def test_the_plan_serialises_to_json():
    payload = json.dumps(plan(config()).as_json())
    assert json.loads(payload)["totals"]["dollars"] > 0


# ── ssh and api argv ──────────────────────────────────────────────────────────

def test_ssh_argv_uses_batch_mode_so_a_missing_key_fails_instead_of_hanging():
    argv = ssh_argv("203.0.113.9", "root", ["true"])
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv  # no interactive password prompt on a unattended run
    assert argv[-2:] == ["--", "true"]


def test_ssh_argv_passes_the_identity_only_when_given():
    assert "-i" not in ssh_argv("h", "root", ["true"])
    assert ssh_argv("h", "root", ["true"], "/k")[-5:-2] == ["-i", "/k", "root@h"]


def test_api_create_uses_the_documented_mi300x_size_slug_and_the_key():
    argv = api_argv("create", "smol-ladder", "AMD_CLOUD_API_TOKEN", "atl1", "gpu-mi300x1-192gb",
                    "img", "aa:bb:cc", "smol-ladder")
    assert "gpu-mi300x1-192gb" in argv
    assert argv[argv.index("--ssh-keys") + 1] == "aa:bb:cc"
    assert "atl1" in argv


def test_api_destroy_is_forced_and_not_power_off():
    # These are different operations with different billing consequences, and conflating them is
    # the whole failure this runbook exists to prevent.
    argv = api_argv("destroy", "d", "T", "atl1", "s", "i", None, "d")
    assert argv[-3:] == ["delete", "d", "--force"]
    off = api_argv("power-off", "d", "T", "atl1", "s", "i", None, "d")
    assert "power-off" in off and "delete" not in off


def test_api_verbs_never_pass_a_token_on_the_command_line():
    for verb in ("create", "destroy", "power-off", "power-on", "get", "list"):
        argv = api_argv(verb, "d", "AMD_CLOUD_API_TOKEN", "atl1", "s", "i", None, "d")
        assert "AMD_CLOUD_API_TOKEN" not in " ".join(argv)
        assert not any("=" in a and a.startswith("DIGITALOCEAN") for a in argv)


def test_an_unknown_api_verb_is_rejected():
    with pytest.raises(ValueError):
        api_argv("reboot", "d", "T", "atl1", "s", "i", None, "d")


# ── the dry run, as the owner actually runs it ────────────────────────────────

def test_dry_run_works_on_a_laptop_with_no_ssh_key_and_no_token(tmp_path):
    """`--dry-run` must never need a credential, a droplet, or the network."""
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--host", "203.0.113.9", "--limit", "10", "--dry-run"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd=str(REPO_ROOT), check=True)
    assert "## dry run" in out.stdout
    assert "ssh" in out.stdout
    assert "poweroff" in out.stdout


def test_dry_run_prints_the_budget_before_the_commands():
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--host", "203.0.113.9", "--dry-run"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, cwd=str(REPO_ROOT), check=True)
    assert out.stdout.index("## budget") < out.stdout.index("## plan")
    assert "$2.59" in out.stdout


def test_api_dry_run_works_with_no_token_and_runs_nothing(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    for flag in ("--destroy", "--power-off", "--status"):
        out = subprocess.run(
            [sys.executable, str(DRIVER), "--mode", "api", "--droplet", "d", flag, "--dry-run"],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
        assert out.returncode == 0, out.stderr
        assert "doctl" in out.stdout


def test_api_mode_without_a_token_refuses_and_says_where_to_put_it(tmp_path):
    # The driver reads the repo's .env by absolute path, so an empty string in the environment is
    # what makes this deterministic: an explicitly-set variable wins over the file, whatever the
    # developer happens to have in .env locally.
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--mode", "api", "--destroy", "--droplet", "d"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "AMD_CLOUD_API_TOKEN": ""},
        cwd=str(tmp_path))
    assert out.returncode != 0
    assert "AMD_CLOUD_API_TOKEN" in out.stderr


def test_a_missing_doctl_is_reported_as_such(tmp_path):
    # The other refusal on the same path, and it needs a token present so it is reachable. Given
    # as an env var rather than a .env so the test never depends on the developer's own file.
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--mode", "api", "--destroy", "--droplet", "d"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "AMD_CLOUD_API_TOKEN": "not-a-real-token"},
        cwd=str(tmp_path))
    assert out.returncode != 0
    assert "doctl" in out.stderr


def test_the_runbooks_table_is_the_drivers_table():
    """docs/AMD_RUNBOOK.md quotes the budget; the two must not be able to drift.

    The doc is prose a human reads before spending money, so a number in it that disagrees with the
    number the tool prints is worse than having no table. Parse the real markdown rather than
    restating the figures, so editing either side breaks this.
    """
    text = (REPO_ROOT / "docs" / "AMD_RUNBOOK.md").read_text()
    # The TOTAL row is the plan's total, which includes the final sync's quarter hour that
    # budget_report's setup row does not carry. Assert against the plan, which is what the table
    # actually enumerates step by step.
    p = plan(config())
    # The TOTAL row: | | **TOTAL** | **~13 h** | **11.00** | **$28.5** | ...
    total = re.search(r"\|\s*\|\s*\*\*TOTAL\*\*\s*\|[^|]*\|\s*\*\*([\d.]+)\*\*\s*\|"
                      r"\s*\*\*(?:~)?\$([\d.]+)\*\*\s*\|", text)
    assert total, "the run plan table has no parseable TOTAL row"
    assert float(total.group(1)) == pytest.approx(p.hours, abs=0.01)
    assert float(total.group(2)) == pytest.approx(p.dollars, abs=0.05)


def test_the_runbook_names_the_official_price_not_the_plans():
    text = (REPO_ROOT / "docs" / "AMD_RUNBOOK.md").read_text()
    assert "$2.59" in text
    # The $1.99 figure is discussed as the unconfirmed alternative, not as the price.
    assert "$1.99" in text
    assert "could not be confirmed" in text


def test_json_output_is_machine_readable():
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--host", "203.0.113.9", "--json", "--dry-run"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, cwd=str(REPO_ROOT), check=True)
    payload = json.loads(out.stdout)
    assert payload["budget"]["price_per_gpu_hour"] == 2.59
    assert payload["plan"]["steps"][0]["name"] == "bootstrap"


def test_an_unknown_arm_is_refused_before_anything_runs():
    out = subprocess.run(
        [sys.executable, str(DRIVER), "--host", "h", "--arms", "A,Z", "--dry-run"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, cwd=str(REPO_ROOT))
    assert out.returncode != 0
    assert "unknown arm" in out.stderr
