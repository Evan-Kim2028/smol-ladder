"""The AMD session's logic: money, order, refusals, resume, serving, evaluation. No network, no
droplet, no GPU, no real clock, no real token.

What is worth being wrong about here is a number, an order, or a refusal: what the plan costs,
what order it runs in, what it will not start, what it reports about money already spent, and what
it destroys. Those are pure functions or small stubbed objects, so they are tested directly. The
shell scripts that matter are run for real against stub binaries (the multi-LoRA serve command, the
watchdog tick); the rest are shellchecked.

Properties this file exists to keep:
  * the numbers in docs/AMD_RUNBOOK.md cannot drift from the numbers the driver prints;
  * a billed step whose projection would pass either cap is REFUSED, tested with projections;
  * the dead-man switch destroys exactly when it should, and never claims to destroy nothing;
  * resume is decided by logic that is exercised against a stubbed Hub and a half-written checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.amd import bench, cloud, deadman, driver, resume, stage  # noqa: E402
from ops.amd import ledger as L  # noqa: E402
from ops.amd import plan as P  # noqa: E402
from ops.amd import probe_tools  # noqa: E402
from ops.amd.doapi import DoApi  # noqa: E402

OPS = REPO_ROOT / "ops" / "amd"
RUNBOOK = REPO_ROOT / "docs" / "AMD_RUNBOOK.md"
HOUR = 3600.0
T0 = 1_800_000_000.0


# ── fixtures and stubs ─────────────────────────────────────────────────────────────

def cfg(**over) -> P.Config:
    c = P.Config(host="203.0.113.9", fingerprint="aa:bb:cc", **over)
    return c


def tokens() -> dict[str, P.SetTokens]:
    return {"A": P.SetTokens(4439, 8_700_000, "test"), "B": P.SetTokens(2029, 6_200_000, "test"),
            "AB": P.SetTokens(6468, 14_900_000, "test")}


def measured(**over) -> P.Measured:
    m = P.Measured(tokens_per_s=9000.0, batch_size=8, grad_accum=1, sec_per_trial=6.0,
                   probe_start_s=240.0, tool_calls_ok=True, checks_ok=True, resume_ok=True)
    for k, v in over.items():
        setattr(m, k, v)
    return m


def names(steps: list[P.Step]) -> list[str]:
    return [s.name for s in steps]


def plan_for(c: P.Config | None = None, m: P.Measured | None = None) -> list[P.Step]:
    return P.build_plan(c or cfg(), tokens(), m or P.Measured())


class FakeApi:
    """The DoApi interface, scripted. Records every call so a test can assert on mutations."""

    def __init__(self, tagged=None, sizes=None, images=None, keys=None):
        self.calls: list[tuple[str, str, object]] = []
        self.tagged = list(tagged or [])
        self.sizes = sizes if sizes is not None else [
            {"slug": P.SIZE_MI350X, "price_hourly": 2.46, "regions": ["ric1"]}]
        self.images = images if images is not None else [
            {"slug": P.IMAGE_DEFAULT, "regions": ["ric1", "tor1"]}]
        self.keys = keys if keys is not None else [{"fingerprint": "aa:bb:cc"}]
        self.create_status = 202
        self.droplet_polls = 0
        self.vanish_after_delete = True
        self.deleted = False
        self.account_extra: list[dict] = []     # droplets on the account that are not ours

    def get(self, path):
        self.calls.append(("GET", path, None))
        if path.startswith("/sizes"):
            return 200, {"sizes": self.sizes}
        if path.startswith("/images"):
            return 200, {"images": self.images}
        if path.startswith("/account/keys"):
            return 200, {"ssh_keys": self.keys}
        if path.startswith("/droplets?tag_name"):
            return 200, {"droplets": [] if (self.deleted and self.vanish_after_delete)
                         else list(self.tagged)}
        if path.startswith("/droplets?per_page"):
            mine = [] if (self.deleted and self.vanish_after_delete) else list(self.tagged)
            return 200, {"droplets": mine + list(self.account_extra)}
        if path.startswith("/droplets/"):
            if self.deleted and self.vanish_after_delete:
                return 404, {"id": "not_found"}
            self.droplet_polls += 1
            status = "new" if self.droplet_polls < 3 else "active"
            nets = {"v4": [{"type": "public", "ip_address": "198.51.100.7"}]} \
                if status == "active" else {"v4": []}
            return 200, {"droplet": {"id": 42, "status": status, "networks": nets}}
        raise AssertionError(path)

    def post(self, path, body):
        self.calls.append(("POST", path, body))
        if self.create_status >= 300:
            return self.create_status, {"error": "nope"}
        return self.create_status, {"droplet": {"id": 42}}

    def delete(self, path):
        self.calls.append(("DELETE", path, None))
        self.deleted = True
        return 204, {}


def mutations(api: FakeApi) -> list:
    return [c for c in api.calls if c[0] in ("POST", "DELETE")]


def ledger_with(tmp_path, *events) -> Path:
    path = tmp_path / "ledger.jsonl"
    for kind, ts, fields in events:
        L.append(path, kind, now=ts, **fields)
    return path


# ═══ the hardware, which changed ═══════════════════════════════════════════════════

def test_the_default_hardware_is_the_spot_gpu_that_can_actually_be_created():
    c = P.Config()
    assert (c.size, c.region, c.price) == ("gpu-mi350x1-288gb-spot", "ric1", 2.46)
    assert P.LISTED_BUT_NOT_OFFERED == "gpu-mi300x1-192gb" and c.size != P.LISTED_BUT_NOT_OFFERED
    assert c.tag == "smol-ladder"


def test_the_fallback_is_on_demand_mi325x_in_tor1_at_3_80():
    assert (P.SIZE_MI325X, P.REGION_MI325X, P.PRICE_MI325X) == ("gpu-mi325x1-256gb", "tor1", 3.80)
    assert P.Config(size=P.SIZE_MI325X).spot is False and P.Config().spot is True


def test_the_default_image_is_the_vllm_one_and_the_reason_is_in_the_source():
    assert P.IMAGE_DEFAULT == "amddeveloperclou-vllm0171"
    assert P.IMAGE_FALLBACK == "amddeveloperclou-pytorch2100rocm7"
    doc = P.__doc__
    assert "pure-Python wheels" in doc and "Why the vLLM image is the default" in doc


def test_the_vllm_floor_is_the_one_qwen35_needs():
    assert P.VLLM_MIN == "0.16.2"
    assert "0.16.2" in (OPS / "smoke.sh").read_text()
    assert "0.16.2" in (OPS / "entrypoint.sh").read_text()


def test_the_default_budgets_are_35_for_the_session_and_90_in_total():
    c = P.Config()
    assert (c.budget, c.total_cap) == (35.0, 90.0) and P.CREDIT == 100.0
    assert P.CREDIT_EXPIRES == "2026-10-18"


def test_the_create_body_attaches_the_key_by_fingerprint_and_tags_the_droplet():
    body = P.create_body(cfg())
    assert body["ssh_keys"] == ["aa:bb:cc"] and body["tags"] == ["smol-ladder"]
    assert (body["region"], body["size"], body["image"]) == ("ric1", P.SIZE_MI350X, P.IMAGE_DEFAULT)
    assert "disk_size_gb" not in body   # not a create parameter; sizes carry their own disk


def test_a_dry_run_with_no_key_shows_a_placeholder_rather_than_inventing_one():
    body = P.create_body(P.Config())
    assert body["ssh_keys"] == ["<ssh-key-fingerprint>"]


# ═══ ledger arithmetic ══════════════════════════════════════════════════════════════

def test_an_empty_ledger_has_spent_nothing():
    s = L.spend([], T0, 2.46)
    assert (s.session, s.total, s.open) == (0.0, 0.0, False)


def test_accrual_counts_created_to_destroyed(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}),
                    (L.DESTROYED, T0 + 2 * HOUR, {}))
    s = L.spend(L.read(p), T0 + 9 * HOUR)
    assert s.total == pytest.approx(4.92) and not s.open


def test_an_open_interval_keeps_accruing_to_the_moment_asked(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))
    s = L.spend(L.read(p), T0 + 1800)
    assert s.total == pytest.approx(1.23) and s.open and s.open_seconds == pytest.approx(1800)


def test_each_interval_is_priced_at_its_own_rate(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}),
                    (L.DESTROYED, T0 + HOUR, {}),
                    (L.CREATED, T0 + 2 * HOUR, {"price_per_hour": 3.80}),
                    (L.DESTROYED, T0 + 3 * HOUR, {}))
    assert L.spend(L.read(p), T0 + 5 * HOUR).total == pytest.approx(2.46 + 3.80)


def test_a_reclaim_closes_the_interval_so_it_stops_accruing(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}),
                    (L.RECLAIMED, T0 + HOUR, {}))
    s = L.spend(L.read(p), T0 + 10 * HOUR)
    assert s.total == pytest.approx(2.46) and not s.open and L.open_interval(L.read(p)) is None


def test_a_create_that_was_never_closed_does_not_silently_stop_billing(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}),
                    (L.CREATED, T0 + HOUR, {"price_per_hour": 2.46}))
    assert L.spend(L.read(p), T0 + 2 * HOUR).total == pytest.approx(4.92)


def test_a_corrupt_or_truncated_line_is_skipped_not_fatal(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))
    with p.open("a") as fh:
        fh.write('{"ts": 1, "event": "destr')   # the process died mid-write
    assert len(L.read(p)) == 1 and L.spend(L.read(p), T0 + HOUR).open


def test_the_ledger_is_append_only_one_line_per_event(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.NOTE, now=1.0, text="a")
    first = p.read_text()
    L.append(p, L.NOTE, now=2.0, text="b")
    assert p.read_text().startswith(first) and len(p.read_text().splitlines()) == 2


def test_only_the_part_of_an_interval_inside_the_session_counts_toward_the_session(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.0}),
                    (L.SESSION, T0 + HOUR, {}), (L.DESTROYED, T0 + 3 * HOUR, {}))
    s = L.spend(L.read(p), T0 + 4 * HOUR)
    assert s.session == pytest.approx(4.0) and s.total == pytest.approx(6.0)


def test_prior_spend_counts_toward_the_total_cap_only(tmp_path):
    p = ledger_with(tmp_path, (L.PRIOR, T0, {"dollars": 40.0}))
    s = L.spend(L.read(p), T0 + HOUR, 2.46)
    assert (s.session, s.total) == (0.0, 40.0)


def test_status_reports_uptime_and_that_a_droplet_is_billing(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 7}),
                    (L.READY, T0 + 60, {"droplet_id": 7, "ip": "198.51.100.7"}))
    s = L.summarise(L.read(p), T0 + 1800, 2.46)
    assert s["uptime_seconds"] == 1800 and s["ip"] == "198.51.100.7" and "RUNNING" in s["state"]
    assert s["session_dollars"] == pytest.approx(1.23)


def test_status_says_nothing_is_billing_once_destroyed(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}), (L.DESTROYED, T0 + 60, {}))
    s = L.summarise(L.read(p), T0 + HOUR, 2.46)
    assert "nothing should be billing" in s["state"] and s["uptime_seconds"] == 0.0


# ═══ the budget gate ════════════════════════════════════════════════════════════════

def test_a_step_that_fits_is_allowed():
    v = L.verdict([], T0, HOUR, 2.46, 35.0, 90.0)
    assert v.allowed and v.session_after == pytest.approx(2.46)


def test_a_step_whose_projection_passes_the_session_budget_is_refused(tmp_path):
    p = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))
    # 12 h already accrued = $29.52; another 3 h = $7.38 more -> $36.90 > $35
    v = L.verdict(L.read(p), T0 + 12 * HOUR, 3 * HOUR, 2.46, 35.0, 90.0)
    assert not v.allowed and "session" in v.reason and "REFUSING" in v.reason
    assert v.session_after == pytest.approx(36.90)


def test_the_total_cap_refuses_even_when_the_session_budget_has_room(tmp_path):
    p = ledger_with(tmp_path, (L.PRIOR, T0, {"dollars": 80.0}))
    v = L.verdict(L.read(p), T0, 5 * HOUR, 2.46, 35.0, 90.0)   # 80 + 12.30 > 90, session 12.30 < 35
    assert not v.allowed and "total" in v.reason


def test_a_projection_exactly_on_the_cap_is_allowed_and_one_cent_over_is_refused():
    on = L.verdict([], T0, 10 * HOUR, 3.5, 35.0, 90.0)
    over = L.verdict([], T0, 10 * HOUR + 10, 3.5, 35.0, 90.0)
    assert on.allowed and not over.allowed


def test_the_reserve_keeps_sync_and_destroy_affordable():
    # 14 h at 2.46 = 34.44; fits alone, but not with the 360 s sync+destroy reserve held back.
    assert L.verdict([], T0, 14 * HOUR, 2.46, 35.0, 90.0).allowed
    assert not L.verdict([], T0, 14 * HOUR + 600, 2.46, 35.0, 90.0, P.RESERVE_S).allowed


def test_the_refusal_names_the_cap_and_the_shortfall():
    v = L.verdict([], T0, 20 * HOUR, 2.46, 35.0, 90.0)
    assert "$35.00" in v.reason and "exceeds" in v.reason


def test_the_driver_gate_raises_instead_of_starting_a_step_that_would_overspend(tmp_path):
    c = cfg(ledger=str(ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))))
    step = P.Step("train", "sft-A", "droplet", seconds=3 * HOUR)
    with pytest.raises(SystemExit) as exc:
        driver.gate(c, step, now=T0 + 12 * HOUR)
    assert "STOPPING BEFORE 'sft-A'" in str(exc.value)


def test_the_driver_gate_lets_an_unbilled_step_through_even_over_budget(tmp_path):
    c = cfg(ledger=str(ledger_with(tmp_path, (L.PRIOR, T0, {"dollars": 999.0}))))
    driver.gate(c, P.Step("sync", "verify-sync", "laptop", billed=False, seconds=0), now=T0)


def test_sync_and_destroy_are_not_refused_by_the_reserve_they_exist_to_spend(tmp_path):
    # 14.3 h in: only the reserve is left. The destroy step must still be allowed to run.
    c = cfg(ledger=str(ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))))
    destroy = next(s for s in plan_for() if s.name == "destroy")
    assert destroy.reserve is False
    driver.gate(c, destroy, now=T0 + 14.1 * HOUR)


def test_the_destroy_and_the_sync_run_even_when_the_session_is_already_over_budget(tmp_path, capsys):
    # 40 h in: $98 accrued, past both caps. Refusing the destroy would keep the meter running.
    c = cfg(ledger=str(ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))))
    for name in ("sync-droplet", "sync-pull", "destroy"):
        driver.gate(c, next(s for s in plan_for() if s.name == name), now=T0 + 40 * HOUR)
    assert "runs anyway" in capsys.readouterr().out
    with pytest.raises(SystemExit):     # while an ordinary step in the same state is refused
        driver.gate(c, next(s for s in plan_for() if s.name == "sft-A"), now=T0 + 40 * HOUR)


# ═══ the costed table ═══════════════════════════════════════════════════════════════

def test_every_row_is_price_times_seconds_and_the_total_is_their_sum():
    rows = P.projection(cfg(), tokens(), P.Measured())
    assert P.total_dollars(rows, 2.46) == pytest.approx(sum(r.seconds for r in rows) / HOUR * 2.46)


def test_an_unmeasured_plan_says_so_in_the_rows_and_in_the_printout(capsys):
    rows = P.projection(cfg(), tokens(), P.Measured())
    assert any("UNMEASURED" in r.basis for r in rows)
    driver.print_table(cfg(), rows)
    assert "UNMEASURED" in capsys.readouterr().out


def test_a_measured_plan_has_no_unmeasured_training_or_evaluation_row():
    rows = P.projection(cfg(), tokens(), measured())
    assert not any("UNMEASURED" in r.basis for r in rows if r.stage.startswith(("sft", "serve")))


def test_training_seconds_come_from_tokens_over_measured_throughput():
    rows = {r.stage: r for r in P.projection(cfg(), tokens(), measured(tokens_per_s=10_000.0))}
    assert rows["sft A"].seconds == pytest.approx(8_700_000 / 10_000.0 * P.SAFETY + P.SFT_OVERHEAD_S)
    assert rows["sft AB"].seconds > rows["sft A"].seconds + rows["sft B"].seconds - 400


def test_a_zero_throughput_is_refused_not_divided_by():
    with pytest.raises(ValueError):
        P.sft_seconds(1_000_000, 0.0)


def test_evaluation_time_scales_with_trials_and_the_measured_seconds_per_trial():
    c = cfg(limit=40, samples=2, late_samples=1)
    one = P.eval_seconds(c, 6.0)
    assert one == pytest.approx((40 * 2 + 40 * 3) * 6.0 * P.CONTENTION
                                + 40 * 6.0 * P.PROGRAM_COST_FACTOR * P.CONTENTION)
    assert P.eval_seconds(c, 12.0) == pytest.approx(2 * one)
    assert P.eval_seconds(cfg(limit=40, rungs=("L1",)), 6.0) < one


def test_the_default_plan_with_placeholders_fits_the_session_budget_it_is_costed_against():
    rows = P.projection(cfg(), tokens(), P.Measured())
    total = P.total_dollars(rows, P.PRICE_MI350X)
    assert total < P.DEFAULT_BUDGET
    assert total < P.TOTAL_CAP


def test_the_fallback_hardware_costs_more_for_the_same_plan():
    spot = P.total_dollars(P.projection(cfg(), tokens(), measured()), 2.46)
    ond = P.total_dollars(P.projection(cfg(price=3.80), tokens(), measured()), 3.80)
    assert ond == pytest.approx(spot * 3.80 / 2.46)


def test_a_missing_dataset_gives_a_zero_second_row_that_says_why():
    rows = P.projection(cfg(), {"A": tokens()["A"]}, measured())
    assert next(r for r in rows if r.stage == "sft B").basis.startswith("NO TOKEN COUNT")


def test_token_counts_staged_at_another_max_length_are_not_trusted(tmp_path):
    f = tmp_path / "tokens.json"
    f.write_text(json.dumps({"max_length": 4096, "sets": {
        "A": {"rows": 1, "trained_tokens": 5}}}))
    assert P.load_tokens(f, 8192) is None and P.load_tokens(f, 4096)["A"].trained_tokens == 5


def test_the_heuristic_over_counts_rather_than_under_counts(tmp_path):
    (tmp_path / "train" / "sft_upstream").mkdir(parents=True)
    (tmp_path / "train" / "sft_upstream" / "train.jsonl").write_text('{"messages": []}\n' * 100)
    (tmp_path / "train" / "ja3_sft.jsonl").write_text('{"messages": []}\n' * 10)
    t = P.heuristic_tokens(tmp_path, 8192)
    assert t["AB"].trained_tokens == t["A"].trained_tokens + t["B"].trained_tokens
    assert "pessimistic" in t["A"].method


def test_the_deadline_is_bounded_by_what_the_budget_buys():
    rows = [P.Row("x", 100 * HOUR, "", False)]
    assert P.default_deadline_minutes(cfg(), rows) == pytest.approx(35.0 / 2.46 * 60, abs=1)


# ═══ step ordering ══════════════════════════════════════════════════════════════════

def test_the_session_runs_in_the_order_the_reviewer_runs_it():
    n = names(plan_for())
    order = ["stage-inputs", "preflight", "deadman", "create", "upload", "bootstrap",
             "smoke-checks", "smoke-pull", "probe-serve", "probe-tunnel", "probe-eval", "go-no-go",
             "sft-A", "sft-B", "sft-AB", "serve", "tunnel", "eval-L1", "eval-L2", "eval-L3",
             "eval-L4", "eval-control", "sync-droplet", "sync-pull", "verify-sync", "tunnel-down",
             "destroy"]
    assert n == order


def test_everything_that_can_happen_before_the_droplet_exists_does():
    steps = plan_for()
    created = names(steps).index("create")
    before = steps[:created]
    assert [s.name for s in before] == ["stage-inputs", "preflight", "deadman"]
    assert all(not s.billed for s in before)


def test_no_step_waits_on_a_human_and_the_only_stop_is_the_go_no_go():
    steps = plan_for()
    assert not any("input(" in " ".join(c.argv) or "read -p" in " ".join(c.argv)
                   for s in steps for c in s.cmds)
    assert "STOPS unless it fits" in next(s for s in steps if s.name == "go-no-go").note


def test_the_smoke_measures_before_any_arm_trains_and_stops_for_the_reviewer():
    n = names(plan_for())
    for arm in ("sft-A", "sft-B", "sft-AB"):
        assert n.index("go-no-go") < n.index(arm)
    assert n.index("probe-serve") < n.index("probe-eval") < n.index("go-no-go")


def test_the_arms_train_in_order_a_then_b_then_ab():
    n = names(plan_for())
    assert n.index("sft-A") < n.index("sft-B") < n.index("sft-AB") < n.index("serve")


def test_serving_happens_once_for_all_four_models_after_all_training():
    steps = plan_for()
    assert [s.name for s in steps if s.name == "serve"] == ["serve"]
    assert names(steps).index("serve") > names(steps).index("sft-AB")


def test_the_sync_is_verified_before_the_destroy_and_the_destroy_is_last():
    n = names(plan_for())
    assert n.index("sync-droplet") < n.index("sync-pull") < n.index("verify-sync") < n.index("destroy")
    assert n[-1] == "destroy"
    assert names(plan_for())[-2] == "tunnel-down"


def test_every_droplet_step_comes_after_the_create_and_before_the_destroy():
    n = names(plan_for())
    for s in plan_for():
        if s.where == "droplet":
            assert n.index("create") < n.index(s.name) < n.index("destroy")


def test_the_destroy_is_an_api_step_that_verifies_by_get():
    d = next(s for s in plan_for() if s.name == "destroy")
    assert d.where == "api" and "GET" in d.api and "must be empty" in d.api


def test_dropping_an_arm_removes_its_training_and_its_evaluation():
    c = cfg(arms=("A", "B"))
    n = names(plan_for(c))
    assert "sft-AB" not in n and "sft-A" in n
    assert not any("amd-ab-2b" in cmd.shell() for s in plan_for(c) for cmd in s.cmds)


def test_a_create_step_is_gated_and_priced():
    create = next(s for s in plan_for() if s.name == "create")
    assert create.billed and create.seconds == P.BOOT_S


# ═══ the laptop-side evaluation ═════════════════════════════════════════════════════

def flat(cmds):
    return [c.shell() for c in cmds]


def eval_steps(c=None):
    return [s for s in plan_for(c) if s.phase == "eval"]


def test_the_harness_runs_on_the_laptop_not_the_droplet():
    assert all(s.where == "laptop" for s in eval_steps())
    assert not any("run_ladder" in " ".join(c.argv) for s in plan_for() if s.where == "droplet"
                   for c in s.cmds)


def test_the_evaluations_point_at_the_tunnel_with_the_non_thinking_template():
    for s in eval_steps():
        for cmd in s.cmds:
            env = dict(cmd.env)
            assert env["SMOL_LADDER_BASE_URL"] == "http://127.0.0.1:8000/v1"
            assert json.loads(env["SMOL_LADDER_CHAT_TEMPLATE_KWARGS"]) == {"enable_thinking": False}


def test_the_protocol_is_the_upstream_bash_agent_for_every_model_including_the_base():
    """The SFT data is `FineEnvs/SmolDataEnvs-sft`: one tool named `bash`, submit to
    /workdir/answer.txt. In this repo that is `run_ladder --agent bash`. The base control uses the
    same protocol so that adapter minus base is the effect of SFT, not of a changed protocol."""
    for rung in ("L1", "L2", "L3", "L4"):
        step = next(s for s in eval_steps() if s.name == f"eval-{rung}")
        assert len(step.cmds) == 4
        for cmd in step.cmds:
            assert cmd.argv[cmd.argv.index("--agent") + 1] == "bash"
            assert cmd.argv[cmd.argv.index("--max-turns") + 1] == "16"


def test_that_flag_really_is_the_bash_tool_protocol_in_this_repo():
    from smol_ladder.upstream import BASH_TOOL
    assert BASH_TOOL[0]["function"]["name"] == "bash"
    assert 'choices=["tools", "program", "bash"]' in (REPO_ROOT / "smol_ladder/run_ladder.py").read_text()
    data = REPO_ROOT / "data/train/sft_upstream/train.jsonl"
    if data.exists():
        row = json.loads(data.open().readline())
        assert [t["function"]["name"] for t in row["tools"]] == ["bash"]


def test_the_base_also_gets_a_one_turn_program_control_at_the_end():
    last = eval_steps()[-1]
    assert last.name == "eval-control" and len(last.cmds) == 1
    argv = last.cmds[0].argv
    assert argv[argv.index("--agent") + 1] == "program" and argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--run-tag") + 1] == "amd1-base-program"


def test_each_arm_has_its_own_run_tag_and_the_model_name_the_server_exposes():
    step = next(s for s in eval_steps() if s.name == "eval-L1")
    got = {}
    for cmd in step.cmds:
        a = cmd.argv
        got[a[a.index("--run-tag") + 1]] = a[a.index("--model") + 1]
    assert got == {"amd1-base": "amd-base-2b", "amd1-a": "amd-a-2b", "amd1-b": "amd-b-2b",
                   "amd1-ab": "amd-ab-2b"}


def test_l1_for_all_four_models_comes_before_any_other_rung_and_the_control_is_last():
    n = [s.name for s in eval_steps()]
    assert n == ["eval-L1", "eval-L2", "eval-L3", "eval-L4", "eval-control"]


def test_l1_gets_two_samples_and_later_rungs_one_and_all_are_flags():
    c = cfg(samples=3, late_samples=2, limit=17, rungs=("L1", "L2"), workers=5)
    steps = {s.name: s for s in eval_steps(c)}
    assert set(steps) == {"eval-L1", "eval-L2", "eval-control"}
    a1, a2 = steps["eval-L1"].cmds[0].argv, steps["eval-L2"].cmds[0].argv
    assert a1[a1.index("--samples") + 1] == "3" and a2[a2.index("--samples") + 1] == "2"
    assert a1[a1.index("--limit") + 1] == "17" and a1[a1.index("--workers") + 1] == "5"
    d = eval_steps()[0].cmds[0].argv
    assert d[d.index("--samples") + 1] == "2" and d[d.index("--limit") + 1] == "250"


def test_every_requested_rung_runs_on_every_task_so_denominators_are_equal():
    for s in eval_steps():
        assert "--no-climb" in s.cmds[0].argv


def test_dropping_the_base_removes_it_and_its_program_control():
    steps = eval_steps(cfg(include_base=False))
    assert "eval-control" not in [s.name for s in steps]
    assert all("amd-base-2b" not in c.shell() for s in steps for c in s.cmds)
    assert len(steps[0].cmds) == 3


def test_the_probe_is_twenty_tasks_of_l1_on_the_base_under_the_same_protocol():
    a = P.probe_cmd(cfg()).argv
    assert a[a.index("--limit") + 1] == "20" and a[a.index("--agent") + 1] == "bash"
    assert a[a.index("--rungs") + 1] == "L1" and a[a.index("--run-tag") + 1] == "amd1-probe"


def test_merged_mode_gives_each_model_its_own_port_and_the_tunnel_forwards_them_all():
    c = cfg(serve_mode="merged")
    assert [P.port_for(c, a) for a in ("base", "A", "B", "AB")] == [8000, 8001, 8002, 8003]
    argv = P.tunnel_up(c).argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "-L"] == [
        f"127.0.0.1:{p}:127.0.0.1:{p}" for p in (8000, 8001, 8002, 8003)]
    cmds = {c_.argv[c_.argv.index("--model") + 1]: dict(c_.env)["SMOL_LADDER_BASE_URL"]
            for c_ in eval_steps(c)[0].cmds}
    assert cmds["amd-ab-2b"].endswith(":8003/v1")


def test_the_tunnel_binds_loopback_fails_fast_and_uses_a_control_socket():
    a = P.tunnel_up(cfg()).argv
    assert "ExitOnForwardFailure=yes" in a and "-N" in a and "-f" in a
    assert "127.0.0.1:8000:127.0.0.1:8000" in a and "-S" in a
    assert list(P.tunnel_down(cfg()).argv[-3:-1]) == ["-O", "exit"]


def test_ssh_is_batch_mode_and_keepalive_so_a_bad_key_fails_instead_of_hanging():
    a = P.ssh(cfg(), ["true"]).argv
    assert "BatchMode=yes" in a and "ServerAliveInterval=30" in a
    assert list(a[-2:]) == ["--", "true"]
    assert "-i" not in a and "-i" in P.ssh(cfg(identity="/k"), ["true"]).argv


# ═══ the multi-LoRA serve command: run the real script against a stub python ═══════

@pytest.fixture
def fake_remote(tmp_path):
    """A droplet-shaped tree: adapters on disk and a stub 'python' that records its argv."""
    root, log = tmp_path / "root", tmp_path / "log"
    for sub in ("a", "b", "ab", "resume_check"):
        d = root / "runs" / (f"sft_{sub}" if sub != "resume_check" else sub)
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_text("x")
    log.mkdir()
    rec = tmp_path / "argv.txt"
    stub = tmp_path / "python-stub"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{rec}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (root / ".syspy").write_text(str(stub) + "\n")
    env = {"PATH": os.environ["PATH"], "AMD_REMOTE_ROOT": str(root), "AMD_REMOTE_LOG": str(log),
           "AMD_HUB_NAMESPACE": "ns", "HOME": str(tmp_path)}
    return env, rec


def run_serve(fake_remote, *args):
    env, rec = fake_remote
    out = subprocess.run(["bash", str(OPS / "serve.sh"), *args], env=env, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for _ in range(50):          # the server is started detached; wait for the stub to run
        if rec.exists() and rec.read_text().strip():
            break
        subprocess.run(["sleep", "0.1"])
    return rec.read_text().splitlines()


def test_serve_sh_starts_ONE_server_with_the_base_and_all_three_adapters(fake_remote):
    argv = run_serve(fake_remote, "--all")
    root = fake_remote[0]["AMD_REMOTE_ROOT"]
    assert argv[:2] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.5-2B"
    assert argv[argv.index("--served-model-name") + 1] == "amd-base-2b"
    assert "--enable-lora" in argv
    assert argv[argv.index("--max-loras") + 1] == "3"
    assert argv[argv.index("--max-lora-rank") + 1] == "16"
    i = argv.index("--lora-modules")
    assert argv[i + 1:i + 4] == [f"amd-a-2b={root}/runs/sft_a", f"amd-b-2b={root}/runs/sft_b",
                                 f"amd-ab-2b={root}/runs/sft_ab"]
    assert argv.count("--model") == 1 and argv.count("--port") == 1   # one server, not four


def test_serve_sh_sets_the_tool_call_parser_the_non_thinking_template_and_binds_loopback(fake_remote):
    argv = run_serve(fake_remote, "--all")
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert json.loads(argv[argv.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"


def test_serve_sh_with_fewer_arms_attaches_fewer_adapters_and_fewer_loras(fake_remote):
    argv = run_serve(fake_remote, "--all", "--arms", "A,B")
    assert argv[argv.index("--max-loras") + 1] == "2"
    assert not any("amd-ab-2b" in a for a in argv)


def test_serve_sh_probe_mode_serves_the_smokes_adapter_under_the_probe_name(fake_remote):
    argv = run_serve(fake_remote, "--probe")
    root = fake_remote[0]["AMD_REMOTE_ROOT"]
    assert argv[argv.index("--lora-modules") + 1] == f"amd-probe-2b={root}/runs/resume_check"
    assert argv[argv.index("--max-loras") + 1] == "1"


def test_merged_mode_serves_the_probe_through_the_same_fallback(fake_remote):
    # The merge step shells out to the venv python (absent here), so only the plan is checked.
    step = next(s for s in plan_for(cfg(serve_mode="merged")) if s.name == "probe-serve")
    assert "--merged" in step.cmds[0].argv
    assert "--merged" not in next(s for s in plan_for() if s.name == "probe-serve").cmds[0].argv


def test_the_tool_call_parser_can_be_switched_without_editing_the_script(fake_remote):
    env, rec = fake_remote
    env["AMD_TOOL_PARSER"] = "hermes"
    argv = run_serve((env, rec), "--all")
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"


def test_serve_names_agree_between_the_scripts_and_the_plan():
    sh = subprocess.run(["bash", "-c", f"source {OPS}/common.sh; for m in base A B AB; do amd_served_name $m; done"],
                        capture_output=True, text=True, env={"PATH": os.environ["PATH"]}).stdout.split()
    assert sh == [P.served_name(m) for m in ("base", "A", "B", "AB")]


# ═══ run_sft.sh: the real script against a stub python ═══════════════════════════

@pytest.fixture
def fake_trainer(tmp_path):
    root, log, venv = tmp_path / "root", tmp_path / "log", tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (root / "data/train/sft_upstream").mkdir(parents=True)
    log.mkdir()
    (root / "data/train/sft_upstream/train.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (root / "data/train/sft_upstream/val.jsonl").write_text('{"a": 3}\n')
    (root / "data/train/ja3_sft.jsonl").write_text('{"b": 1}\n')
    rec = tmp_path / "trainer-argv.txt"
    stub = venv / "bin" / "python"
    stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        case "$*" in
          *ops.amd.resume*) echo "{{\\"state\\": \\"${{STUB_STATE:-fresh}}\\", \\"step\\": null}}" ;;
          *train.sft_lora*) printf '%s\\n' "$@" > "{rec}" ;;
          *) cat > /dev/null ;;
        esac
        """))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": os.environ["PATH"], "AMD_REMOTE_ROOT": str(root), "AMD_REMOTE_LOG": str(log),
           "AMD_VENV": str(venv), "AMD_HUB_NAMESPACE": "ns", "HOME": str(tmp_path)}
    return env, rec, root, log


def run_sft(fake_trainer, *args, state="fresh"):
    env, rec, _, _ = fake_trainer
    out = subprocess.run(["bash", str(OPS / "run_sft.sh"), *args], env={**env, "STUB_STATE": state},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return rec.read_text().splitlines() if rec.exists() else None


def test_run_sft_trains_arm_a_from_the_export_directory_with_resume_and_a_private_hub_repo(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "A")
    root = fake_trainer[0]["AMD_REMOTE_ROOT"]
    assert argv[:2] == ["-m", "train.sft_lora"]
    assert argv[argv.index("--data") + 1] == f"{root}/data/train/sft_upstream"
    assert argv[argv.index("--hub-model-id") + 1] == "ns/smol-ladder-sft-a" and "--resume" in argv
    assert argv[argv.index("--protocol") + 1] == "bash" and argv[argv.index("--precision") + 1] == "bf16"
    assert argv[argv.index("--max-length") + 1] == "8192"


def test_run_sft_uses_the_batch_size_the_smoke_measured(fake_trainer):
    _, _, _, log = fake_trainer
    (log / "measurements.json").write_text(json.dumps(
        {"best": {"per_device_batch_size": 4, "grad_accum": 2, "tokens_per_s": 9000}}))
    argv = run_sft(fake_trainer, "--arm", "A")
    assert argv[argv.index("--batch-size") + 1] == "4" and argv[argv.index("--grad-accum") + 1] == "2"


def test_run_sft_without_a_measurement_still_never_trains_at_batch_one_and_keeps_effective_batch_8(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "A")
    b, a = int(argv[argv.index("--batch-size") + 1]), int(argv[argv.index("--grad-accum") + 1])
    assert b > 1 and b * a == 8


def test_run_sft_arm_b_trains_on_the_single_file(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "B")
    assert argv[argv.index("--data") + 1].endswith("data/train/ja3_sft.jsonl")
    assert argv[argv.index("--hub-model-id") + 1] == "ns/smol-ladder-sft-b"


def test_run_sft_arm_ab_is_the_concatenation_of_a_and_b_with_a_val_set(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "AB")
    d = Path(argv[argv.index("--data") + 1])
    assert d.name == "sft_ab" and len((d / "train.jsonl").read_text().splitlines()) == 3
    assert (d / "val.jsonl").exists()


def test_run_sft_skips_an_arm_that_is_already_finished(fake_trainer):
    assert run_sft(fake_trainer, "--arm", "A", state="done") is None


def test_every_flag_the_scripts_pass_to_the_trainer_exists_in_its_parser():
    helptext = subprocess.run([sys.executable, "-m", "train.sft_lora", "--help"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout
    used = set()
    for path in (OPS / "run_sft.sh", OPS / "resume_check.sh", OPS / "bench.py"):
        text = path.read_text()
        for block in re.findall(r"train\.sft_lora(.*?)(?:\n\n|\)\n)", text, re.S):
            used |= set(re.findall(r'"?(--[a-z][a-z-]+)', block))
    assert {"--data", "--hub-model-id", "--resume", "--protocol", "--precision", "--batch-size",
            "--grad-accum", "--max-length", "--seed", "--max-steps", "--logging-steps"} <= used
    missing = {f for f in used if f not in helptext}
    assert not missing, missing


# ═══ resume ═════════════════════════════════════════════════════════════════════════

def write_safetensors(path: Path, truncate: int = 0) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    blob = struct.pack("<Q", len(header)) + header + b"\0" * 8
    path.write_bytes(blob[:-truncate] if truncate else blob)


def make_checkpoint(out: Path, step: int, complete: bool = True, truncated: bool = False) -> Path:
    d = out / f"checkpoint-{step}"
    d.mkdir(parents=True)
    write_safetensors(d / "adapter_model.safetensors", truncate=4 if truncated else 0)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (d / "optimizer.pt").write_bytes(b"o")
    (d / "scheduler.pt").write_bytes(b"s")
    if complete:
        (d / "rng_state.pth").write_bytes(b"r")
    return d


class HubStub:
    """Stands in for the private Hub repo: serves a `last-checkpoint/` folder."""

    def __init__(self, step: int | None, tmp_path: Path):
        self.step, self.tmp, self.downloads = step, tmp_path, 0

    def list_files(self, repo):
        return ["last-checkpoint/trainer_state.json"] if self.step is not None else ["README.md"]

    def download(self, repo, subfolder, dest):
        self.downloads += 1
        src = Path(dest) / subfolder
        src.mkdir(parents=True)
        write_safetensors(src / "adapter_model.safetensors")
        (src / "trainer_state.json").write_text(json.dumps({"global_step": self.step}))
        for n in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (src / n).write_bytes(b"x")
        return src


def test_a_fresh_start_is_fresh(tmp_path):
    assert resume.status(tmp_path / "run")["state"] == "fresh"


def test_a_complete_local_checkpoint_is_resumed_from_disk(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    make_checkpoint(out, 100)
    make_checkpoint(out, 200)
    assert resume.status(out) == {"state": "resume-local", "step": 200, "pruned": []}


def test_a_checkpoint_half_written_at_the_kill_is_set_aside_not_resumed_from(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    make_checkpoint(out, 100)
    make_checkpoint(out, 200, complete=False)       # killed mid-save: no rng_state yet
    res = resume.status(out)
    assert res["state"] == "resume-local" and res["step"] == 100 and res["pruned"] == ["checkpoint-200"]
    assert (out / "partial-200").is_dir() and not (out / "checkpoint-200").exists()


def test_a_truncated_adapter_file_makes_a_checkpoint_incomplete(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    make_checkpoint(out, 100, truncated=True)
    assert not resume.is_complete(out / "checkpoint-100")
    assert resume.status(out)["state"] == "fresh"


def test_on_a_fresh_droplet_the_latest_checkpoint_comes_back_from_the_hub(tmp_path):
    out = tmp_path / "run"
    hub = HubStub(300, tmp_path)
    res = resume.status(out, "ns/repo", hub)
    assert res == {"state": "resume-hub", "step": 300, "pruned": []}
    assert resume.is_complete(out / "checkpoint-300") and hub.downloads == 1


def test_a_local_checkpoint_wins_and_the_hub_is_not_touched(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    make_checkpoint(out, 100)
    hub = HubStub(300, tmp_path)
    assert resume.status(out, "ns/repo", hub)["state"] == "resume-local" and hub.downloads == 0


def test_a_hub_repo_with_no_checkpoint_means_a_fresh_start(tmp_path):
    assert resume.status(tmp_path / "run", "ns/repo", HubStub(None, tmp_path))["state"] == "fresh"


def test_a_finished_arm_is_skipped(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    write_safetensors(out / "adapter_model.safetensors")
    (out / "adapter_config.json").write_text("{}")
    (out / resume.DONE).write_text("done")
    assert resume.status(out)["state"] == "done"


def test_a_done_marker_without_a_readable_adapter_does_not_count(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / resume.DONE).write_text("done")
    assert resume.status(out)["state"] == "fresh"


def test_the_resume_verdict_demands_the_checkpoint_the_kill_left():
    log = "1000 train rows\nresuming from /opt/smol-ladder/runs/resume_check/checkpoint-50\n"
    assert resume.verdict(50, log)["ok"] is True


def test_a_resume_that_starts_from_zero_is_a_failure_not_a_resume():
    v = resume.verdict(50, "no checkpoint to resume from; starting fresh\n")
    assert v["ok"] is False and v["resumed_from"] is None


def test_a_resume_from_the_wrong_checkpoint_fails():
    assert resume.verdict(100, "resuming from x/checkpoint-50")["ok"] is False


def test_the_trainer_prints_the_message_the_verdict_parses():
    assert "resuming from {checkpoints[-1]}" in (REPO_ROOT / "train/sft_lora.py").read_text()


# ═══ benchmark arithmetic and parsing ═══════════════════════════════════════════════

def test_steps_per_second_ignores_the_warmup_steps():
    # three slow warmup steps (compilation) then a steady 2 steps/s
    stamps = [0.0, 5.0, 10.0] + [10.5 + 0.5 * i for i in range(10)]
    assert bench.steps_per_second(stamps) == pytest.approx(2.0)


def test_too_few_steady_steps_is_not_a_measurement():
    assert bench.steps_per_second([0, 1, 2, 3, 4]) is None


def test_tokens_per_second_is_steps_times_effective_batch_times_real_tokens_per_row():
    assert bench.tokens_per_second(0.5, 8, 2000.0) == pytest.approx(8000.0)


def test_the_best_config_is_the_fastest_one_that_did_not_run_out_of_memory():
    configs = [{"per_device_batch_size": 2, "ok": True, "tokens_per_s": 5000},
               {"per_device_batch_size": 4, "ok": True, "tokens_per_s": 7000},
               {"per_device_batch_size": 8, "ok": False, "oom": True, "tokens_per_s": 9000}]
    assert bench.choose_best(configs)["per_device_batch_size"] == 4
    assert bench.choose_best([{"ok": False}]) is None


def test_the_benchmark_never_tries_batch_one_and_holds_the_effective_batch_at_eight():
    assert 1 not in P.BENCH_BATCHES and P.EFFECTIVE_BATCH == 8
    assert all(P.EFFECTIVE_BATCH % b == 0 for b in P.BENCH_BATCHES)


STEP_PRINTER = textwrap.dedent("""
    import sys, time
    for i in range(40):
        print({"loss": "1.0", "epoch": str(i)}, flush=True)
        time.sleep(0.02)
    """)


def test_run_config_times_a_real_subprocess_of_per_step_log_lines():
    r = bench.run_config([sys.executable, "-c", STEP_PRINTER], seconds=0.5, mean_tokens=1000.0,
                         bs=4, accum=2)
    assert r["ok"] and r["steps_per_s"] > 5 and r["tokens_per_s"] == pytest.approx(
        r["steps_per_s"] * 8 * 1000.0, rel=0.01)


def test_run_config_flags_an_out_of_memory_run_instead_of_reporting_its_speed():
    code = 'import sys; print("torch.OutOfMemoryError: HIP out of memory"); sys.exit(1)'
    r = bench.run_config([sys.executable, "-c", code], seconds=0.2, mean_tokens=1.0, bs=8, accum=1)
    assert r["oom"] and not r["ok"]


def test_a_hung_benchmark_process_is_killed_rather_than_waited_on_while_billing():
    r = bench.run_config([sys.executable, "-c", "import time; time.sleep(60)"], seconds=1.0,
                         mean_tokens=1.0, bs=2, accum=4, hard_timeout=1.0)
    assert r["timed_out"] and not r["ok"]


def test_the_smoke_report_is_ok_only_if_every_check_passed():
    ok = [{"ok": True, "name": "a"}, {"ok": True, "name": "b"}]
    bad = ok + [{"ok": False, "name": "c"}]
    best = {"per_device_batch_size": 4, "grad_accum": 2, "tokens_per_s": 7000.0}
    assert bench.finalize(ok, {"best": best, "configs": []}, {"ok": True})["checks_ok"] is True
    assert bench.finalize(bad, {"best": best, "configs": []}, {"ok": True})["checks_ok"] is False
    assert bench.finalize([], None, None)["checks_ok"] is False


def test_checks_tsv_round_trips(tmp_path):
    f = tmp_path / "c.tsv"
    f.write_text("PASS\t/dev/kfd present\nFAIL\tvllm too old\n")
    assert bench.read_checks(f) == [{"ok": True, "name": "/dev/kfd present"},
                                    {"ok": False, "name": "vllm too old"}]


def test_measurements_become_a_ledger_event_the_plan_reads_back(tmp_path):
    report = {"best": {"tokens_per_s": 9100.0, "per_device_batch_size": 4, "grad_accum": 2},
              "checks_ok": True, "resume": {"ok": True}}
    p = tmp_path / "l.jsonl"
    L.append(p, L.MEASURED, **driver.parse_measurements(report))
    L.append(p, L.MEASURED, **driver.parse_probe_serve("READY_AFTER_S=212\nTOOL_CALLS_OK=1\n"))
    L.append(p, L.MEASURED, sec_per_trial=5.5)
    m = P.measured_from_ledger(L.read(p))
    assert (m.tokens_per_s, m.batch_size, m.grad_accum) == (9100.0, 4, 2)
    assert (m.probe_start_s, m.tool_calls_ok, m.sec_per_trial, m.resume_ok) == (212.0, True, 5.5, True)


def test_seconds_per_trial_comes_from_the_harness_progress_lines():
    out = "250 tasks...\n[1/20] t1 L1 s0 reward=1.0 pred='3' exit 0\n[2/20] t2 L1 s0 reward=0.0 pred='' exit 0\n"
    assert driver.seconds_per_trial(out, 30.0) == pytest.approx(15.0)
    assert driver.seconds_per_trial("nothing ran", 30.0) is None


# ═══ go / no-go ═════════════════════════════════════════════════════════════════════

def events_for(tmp_path, **meas) -> tuple[P.Config, list[dict]]:
    c = cfg(ledger=str(tmp_path / "l.jsonl"), stage_dir=str(tmp_path / "stage"))
    (tmp_path / "stage").mkdir()
    (tmp_path / "stage" / "tokens.json").write_text(json.dumps({
        "max_length": 8192, "sets": {k: {"rows": v.rows, "trained_tokens": v.trained_tokens}
                                     for k, v in tokens().items()}}))
    fields = {"tokens_per_s": 9000.0, "batch_size": 8, "grad_accum": 1, "sec_per_trial": 6.0,
              "probe_start_s": 240.0, "tool_calls_ok": True, "checks_ok": True, "resume_ok": True}
    fields.update(meas)
    L.append(c.ledger, L.MEASURED, now=T0, **fields)
    return c, L.read(Path(c.ledger))


def test_a_clean_smoke_that_fits_the_budget_is_a_go(tmp_path):
    c, ev = events_for(tmp_path)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert ok and reasons == []


def test_a_failed_checklist_item_is_a_no_go(tmp_path):
    c, ev = events_for(tmp_path, checks_ok=False)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("checklist" in r and "FAILED" in r for r in reasons)


def test_empty_tool_calls_are_a_no_go_before_a_cent_goes_on_training(tmp_path):
    c, ev = events_for(tmp_path, tool_calls_ok=False)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("tool calls" in r for r in reasons)


def test_a_failed_resume_is_a_no_go(tmp_path):
    c, ev = events_for(tmp_path, resume_ok=False)
    assert not driver.go_no_go(c, ev, T0)[0]


def test_an_unmeasured_throughput_is_a_no_go_not_a_guess(tmp_path):
    c, ev = events_for(tmp_path, tokens_per_s=None)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("throughput" in r for r in reasons)


def test_a_projection_past_the_session_budget_stops_for_the_reviewer(tmp_path):
    c, ev = events_for(tmp_path, sec_per_trial=60.0)       # evaluation alone is then enormous
    ok, reasons, info = driver.go_no_go(c, ev, T0)
    assert not ok and any("exceeds the $35.00 budget" in r for r in reasons)
    assert info["remaining"] > 35.0


def test_money_already_spent_counts_toward_the_projection(tmp_path):
    c, ev = events_for(tmp_path)
    L.append(c.ledger, L.CREATED, now=T0, price_per_hour=2.46)
    ev = L.read(Path(c.ledger))
    ok_early, _, _ = driver.go_no_go(c, ev, T0 + 600)
    ok_late, reasons, _ = driver.go_no_go(c, ev, T0 + 12 * HOUR)
    assert ok_early and not ok_late and any("spent" in r for r in reasons)


def test_a_no_go_is_recorded_and_train_refuses_to_run_without_a_go(tmp_path, capsys):
    c, ev = events_for(tmp_path, checks_ok=False)
    assert driver.cmd_project(c, ev) is False
    assert driver.last_go(L.read(Path(c.ledger))) is False
    assert "NO-GO" in capsys.readouterr().out


def test_a_go_prints_the_command_that_re_arms_the_deadman_from_measurements(tmp_path, capsys):
    c, ev = events_for(tmp_path)
    assert driver.cmd_project(c, ev) is True
    out = capsys.readouterr().out
    assert "GO/NO-GO: GO" in out and "ops/amd/deadman.py --deadline-minutes" in out


# ═══ the dead-man switch ════════════════════════════════════════════════════════════

def droplet(status="active", name="smol-ladder", created=T0, price=2.46):
    return {"id": 1, "name": name, "status": status,
            "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(created)),
            "size": {"price_hourly": price}}


def test_the_deadman_does_nothing_while_the_clock_and_the_cost_are_under():
    d = deadman.decide(T0, T0 + HOUR, 5.0, 5.0, 35.0, 90.0, [droplet()])
    assert not d.destroy and d.reason.startswith("ok")


def test_the_deadman_destroys_at_the_wall_clock_deadline():
    d = deadman.decide(T0 + HOUR, T0 + HOUR, 1.0, 1.0, 35.0, 90.0, [droplet()])
    assert d.destroy and "deadline" in d.reason


def test_the_deadman_destroys_when_the_session_cost_reaches_the_cap():
    d = deadman.decide(T0, T0 + 99 * HOUR, 35.0, 35.0, 35.0, 90.0, [droplet()])
    assert d.destroy and "session cost" in d.reason


def test_the_deadman_destroys_when_the_total_reaches_the_total_cap():
    d = deadman.decide(T0, T0 + 99 * HOUR, 5.0, 90.0, 35.0, 90.0, [droplet()])
    assert d.destroy and "total cost" in d.reason


def test_the_deadman_destroys_a_powered_off_droplet_because_power_off_still_bills():
    d = deadman.decide(T0, T0 + 99 * HOUR, 1.0, 1.0, 35.0, 90.0, [droplet(status="off")])
    assert d.destroy and "powered off" in d.reason


def test_the_deadman_never_claims_to_destroy_a_droplet_that_does_not_exist():
    d = deadman.decide(T0 + 99 * HOUR, T0, 99.0, 99.0, 35.0, 90.0, [])
    assert not d.destroy and "nothing to destroy" in d.reason


def test_the_deadline_fires_even_when_the_ledger_is_empty_and_wrong():
    d = deadman.decide(T0 + 2 * HOUR, T0 + HOUR, 0.0, 0.0, 35.0, 90.0, [droplet()])
    assert d.destroy


def test_tick_destroys_by_tag_verifies_with_a_get_and_logs_to_the_ledger(tmp_path):
    api = FakeApi(tagged=[droplet()])
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    d = deadman.tick(api, "smol-ladder", led, T0 + 2 * HOUR, T0 + HOUR, 35.0, 90.0, 2.46,
                     dry_run=False, sleep=lambda s: None)
    assert d.destroy
    assert mutations(api) == [("DELETE", "/droplets?tag_name=smol-ladder", None)]
    after = api.calls[api.calls.index(("DELETE", "/droplets?tag_name=smol-ladder", None)) + 1:]
    assert any(c[0] == "GET" and c[1].startswith("/droplets?tag_name=smol-ladder") for c in after)
    events = L.read(led)
    assert L.open_interval(events) is None
    assert any(e["event"] == L.NOTE and "deadman destroyed" in e.get("text", "") for e in events)


def test_a_dry_run_tick_decides_and_prints_but_sends_nothing(tmp_path, capsys):
    api = FakeApi(tagged=[droplet()])
    d = deadman.tick(api, "smol-ladder", tmp_path / "l.jsonl", T0 + 2 * HOUR, T0 + HOUR, 35.0,
                     90.0, 2.46, dry_run=True, sleep=lambda s: None)
    assert d.destroy and mutations(api) == [] and "DRY RUN" in capsys.readouterr().out


def test_tick_prices_an_unrecorded_create_from_the_apis_own_age_and_hourly_rate(tmp_path):
    # The ledger knows nothing (the create was never recorded) but the API shows a droplet that
    # has been up 15 h at $2.46/h = $36.90: over the session cap.
    api = FakeApi(tagged=[droplet(created=T0)])
    d = deadman.tick(api, "smol-ladder", tmp_path / "empty.jsonl", T0 + 15 * HOUR,
                     T0 + 99 * HOUR, 35.0, 90.0, 2.46, dry_run=True, sleep=lambda s: None)
    assert d.destroy and "session cost" in d.reason


def test_tick_notices_a_spot_reclaim_and_stops_the_ledger_accruing(tmp_path):
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    api = FakeApi(tagged=[])      # the API says nothing with the tag exists
    d = deadman.tick(api, "smol-ladder", led, T0 + HOUR, T0 + 99 * HOUR, 35.0, 90.0, 2.46,
                     dry_run=False, sleep=lambda s: None)
    assert not d.destroy and mutations(api) == []
    events = L.read(led)
    assert events[-1]["event"] == L.RECLAIMED and L.open_interval(events) is None
    assert L.spend(events, T0 + 50 * HOUR).total == pytest.approx(2.46)


def test_a_just_created_droplet_missing_from_the_tag_listing_is_not_mistaken_for_a_reclaim(tmp_path):
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    assert cloud.reconcile([], led, T0 + 30) is False and L.open_interval(L.read(led)) is not None
    assert cloud.reconcile([], led, T0 + 300) is True


def test_the_deadman_decision_is_a_pure_function_of_its_inputs():
    args = (T0, T0 + HOUR, 1.0, 1.0, 35.0, 90.0, [droplet()])
    assert deadman.decide(*args) == deadman.decide(*args)


# ═══ the cloud lifecycle against a fake API ═════════════════════════════════════════

def test_preflight_passes_when_everything_the_plan_needs_exists():
    res = cloud.preflight(FakeApi(), cfg())
    assert all(ok for _, ok, _ in res), res
    assert {n for n, _, _ in res} >= {"size offered in region", "image slug exists",
                                      "ssh key fingerprint registered", "no droplet already tagged"}


def test_preflight_fails_if_the_size_is_not_offered_in_the_region():
    api = FakeApi(sizes=[{"slug": P.SIZE_MI350X, "price_hourly": 2.46, "regions": []}])
    assert not dict((n, ok) for n, ok, _ in cloud.preflight(api, cfg()))["size offered in region"]


def test_preflight_fails_if_the_ssh_key_is_not_registered_on_the_account():
    assert not dict((n, ok) for n, ok, _ in cloud.preflight(FakeApi(keys=[]), cfg()))[
        "ssh key fingerprint registered"]


def test_preflight_fails_if_the_price_moved():
    api = FakeApi(sizes=[{"slug": P.SIZE_MI350X, "price_hourly": 3.1, "regions": ["ric1"]}])
    assert not dict((n, ok) for n, ok, _ in cloud.preflight(api, cfg()))["price matches the plan"]


def test_preflight_fails_if_a_tagged_droplet_already_exists():
    res = dict((n, ok) for n, ok, _ in cloud.preflight(FakeApi(tagged=[droplet()]), cfg()))
    assert not res["no droplet already tagged"]


def test_preflight_only_ever_reads():
    api = FakeApi()
    cloud.preflight(api, cfg())
    assert mutations(api) == []


def test_create_records_the_interval_before_sending_then_the_ip_once_active(tmp_path):
    api, led = FakeApi(), tmp_path / "l.jsonl"
    info = cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0)
    assert info == {"droplet_id": 42, "ip": "198.51.100.7"}
    kinds = [e["event"] for e in L.read(led)]
    assert kinds[:2] == [L.SESSION, L.CREATED] and kinds[-1] == L.READY
    post = next(c for c in api.calls if c[0] == "POST")
    assert post[1] == "/droplets" and post[2]["tags"] == ["smol-ladder"]
    assert post[2]["ssh_keys"] == ["aa:bb:cc"]
    assert L.spend(L.read(led), T0 + HOUR).open


def test_recreating_after_a_reclaim_is_the_same_session_so_the_spend_still_counts(tmp_path):
    led = tmp_path / "l.jsonl"
    cloud.create(FakeApi(), cfg(), led, sleep=lambda s: None, now=lambda: T0)
    L.append(led, L.RECLAIMED, now=T0 + HOUR)
    cloud.create(FakeApi(), cfg(), led, sleep=lambda s: None, now=lambda: T0 + 2 * HOUR)
    ev = L.read(led)
    assert [e["event"] for e in ev].count(L.SESSION) == 1
    assert L.spend(ev, T0 + 2 * HOUR).session == pytest.approx(2.46)          # the first hour still counts
    L.append(led, L.DESTROYED, now=T0 + 2.5 * HOUR)
    cloud.create(FakeApi(), cfg(), led, sleep=lambda s: None, now=lambda: T0 + 3 * HOUR, new_session=True)
    assert [e["event"] for e in L.read(led)].count(L.SESSION) == 2


def test_a_failed_create_closes_the_pending_interval_so_nothing_phantom_bills(tmp_path):
    api, led = FakeApi(), tmp_path / "l.jsonl"
    api.create_status = 422
    with pytest.raises(SystemExit):
        cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0)
    assert L.open_interval(L.read(led)) is None


def test_destroy_closes_the_ledger_only_after_a_follow_up_get_shows_nothing_left(tmp_path):
    api = FakeApi(tagged=[droplet()])
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))
    assert cloud.destroy(api, "smol-ladder", led, sleep=lambda s: None, now=lambda: T0 + HOUR)
    assert api.calls[0] == ("DELETE", "/droplets?tag_name=smol-ladder", None)
    assert api.calls[1][0] == "GET" and L.read(led)[-1]["verified_gone"] is True
    assert L.open_interval(L.read(led)) is None


def test_destroy_that_cannot_be_verified_leaves_the_interval_open_and_says_so(tmp_path):
    api = FakeApi(tagged=[droplet()])
    api.vanish_after_delete = False
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46}))
    assert cloud.destroy(api, "smol-ladder", led, sleep=lambda s: None, checks=3,
                         now=lambda: T0 + HOUR) is False
    assert L.open_interval(L.read(led)) is not None
    assert "NOT verified" in L.read(led)[-1]["text"]


def test_the_api_client_never_leaks_its_token():
    c = DoApi("super-secret-token")
    assert "super-secret-token" not in repr(c) and "super-secret-token" not in str(c.__dict__.get("_base"))
    with pytest.raises(SystemExit):
        DoApi("")


# ═══ verify-sync ════════════════════════════════════════════════════════════════════

def make_runs(root: Path, tags, n=2):
    for tag in tags:
        for i in range(n):
            d = root / tag / "test" / f"task{i}" / "L1"
            d.mkdir(parents=True)
            (d / "result.json").write_text("{}")


def test_verify_sync_passes_when_adapters_checksums_and_runs_are_all_present(tmp_path, capsys):
    c = cfg(local_logs=str(tmp_path / "logs"))
    (tmp_path / "logs" / "adapters" / "A").mkdir(parents=True)
    f = tmp_path / "logs" / "adapters" / "A" / "adapter_model.safetensors"
    f.write_bytes(b"abc")
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text(f"{stage.sha256_file(f)}  adapters/A/adapter_model.safetensors\n")
    make_runs(tmp_path / "runs", P.expected_trials(c))
    hub = lambda repo: ["adapter_config.json", "adapter_model.safetensors"]  # noqa: E731
    assert driver.verify_sync(c, hub_files=hub, runs_root=tmp_path / "runs", namespace="ns") is True


def test_verify_sync_fails_if_an_adapter_is_not_readable_on_the_hub(tmp_path):
    c = cfg(local_logs=str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("")
    make_runs(tmp_path / "runs", P.expected_trials(c))
    assert driver.verify_sync(c, hub_files=lambda r: [], runs_root=tmp_path / "runs",
                              namespace="ns") is False


def test_verify_sync_fails_on_a_checksum_mismatch_and_on_a_missing_run_tag(tmp_path):
    c = cfg(local_logs=str(tmp_path / "logs"))
    (tmp_path / "logs" / "adapters").mkdir(parents=True)
    (tmp_path / "logs" / "adapters" / "x").write_bytes(b"changed")
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("0" * 64 + "  adapters/x\n")
    assert driver.sha_check(tmp_path / "logs") == ["adapters/x"]
    hub = lambda repo: ["adapter_config.json", "adapter_model.safetensors"]  # noqa: E731
    assert driver.verify_sync(c, hub_files=hub, runs_root=tmp_path / "no-runs", namespace="ns") is False


# ═══ staging, everything before the droplet exists ══════════════════════════════════

def make_repo(tmp_path: Path, with_required: bool = True) -> Path:
    repo = tmp_path / "repo"
    for rel in stage.REQUIRED_IN_COMMIT if with_required else ("README.md",):
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("echo hi\n" if rel.endswith(".sh") else "x = 1\n")
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "add", "-A"])
    subprocess.check_call(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "commit", "-qm", "init"])
    return repo


def make_data(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "train"
    (d / "sft_upstream").mkdir(parents=True)
    row = json.dumps({"messages": [{"role": "user", "content": "hello world " * 10}], "tools": []})
    (d / "sft_upstream" / "train.jsonl").write_text((row + "\n") * 5)
    (d / "sft_upstream" / "val.jsonl").write_text(row + "\n")
    (d / "ja3_sft.jsonl").write_text((row + "\n") * 3)
    (d / "ja3_sft.manifest.json").write_text("{}")
    return tmp_path / "data"


def test_staging_builds_verified_tarballs_the_pinned_commit_and_a_private_secrets_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    repo, data, out = make_repo(tmp_path), make_data(tmp_path), tmp_path / "stage"
    info = stage.stage(out, "HEAD", 8192, data, repo=repo, namespace="ns", encode=len)
    for name in ("code.tar.gz", "sft_a.tar.gz", "sft_b.tar.gz", "tokens.json", "repo.txt",
                 "SHA256SUMS", "entrypoint.sh", "remote.env"):
        assert (out / name).exists(), name
    assert (out / "repo.txt").read_text().strip() == info["commit"] == subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert stage.verify_sums(out) == []
    assert "remote.env" not in (out / "SHA256SUMS").read_text()
    mode = stat.S_IMODE((out / "remote.env").stat().st_mode)
    assert mode == 0o600
    assert "HF_TOKEN=hf_secret" in (out / "remote.env").read_text()
    assert "hf_secret" not in " ".join(str(v) for v in info.values())
    tok = json.loads((out / "tokens.json").read_text())
    assert tok["max_length"] == 8192 and tok["sets"]["AB"]["rows"] == 8


def test_a_corrupted_tarball_is_caught_by_the_checksums(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    out = tmp_path / "stage"
    stage.stage(out, "HEAD", 8192, make_data(tmp_path), repo=make_repo(tmp_path), namespace="ns")
    with (out / "sft_b.tar.gz").open("ab") as fh:
        fh.write(b"junk")
    assert stage.verify_sums(out) == ["sft_b.tar.gz"]


def test_staging_refuses_a_commit_that_lacks_the_code_the_droplet_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(SystemExit) as exc:
        stage.stage(tmp_path / "s", "HEAD", 8192, make_data(tmp_path),
                    repo=make_repo(tmp_path, with_required=False), namespace="ns")
    assert "lacks" in str(exc.value)


def test_staging_refuses_uncommitted_changes_to_the_code_it_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    repo = make_repo(tmp_path)
    (repo / "ops/amd/smoke.sh").write_text("changed\n")
    with pytest.raises(SystemExit) as exc:
        stage.stage(tmp_path / "s", "HEAD", 8192, make_data(tmp_path), repo=repo, namespace="ns")
    assert "uncommitted" in str(exc.value)


def test_staging_refuses_without_an_hf_token_because_the_hub_is_the_only_durable_copy(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        stage.stage(tmp_path / "s", "HEAD", 8192, make_data(tmp_path), repo=make_repo(tmp_path),
                    namespace="ns")
    assert "HF_TOKEN" in str(exc.value)


def test_token_counting_truncates_each_row_at_the_training_window(tmp_path):
    f = tmp_path / "a.jsonl"
    long_row = {"messages": [{"role": "user", "content": "x" * 100}]}
    f.write_text(json.dumps(long_row) + "\n")
    c = stage.count_file(f, 50, encode=len)
    assert c["raw_tokens"] == 100 + stage.MSG_OVERHEAD_TOKENS and c["trained_tokens"] == 50


def test_without_a_tokenizer_the_counts_say_they_are_an_estimate(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text('{"messages": []}\n' * 10)
    t = stage.build_token_counts([f], f, 8192, encode=None)
    assert "heuristic" in t["sets"]["A"]["method"] and t["sets"]["AB"]["rows"] == 20


# ═══ the CLI, as the reviewer runs it ═══════════════════════════════════════════════

def run_driver(tmp_path, *args):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "AMD_OFFLINE": "1"}
    return subprocess.run([sys.executable, str(OPS / "driver.py"), *args, "--ledger",
                           str(tmp_path / "l.jsonl"), "--stage-dir", str(tmp_path / "none")],
                          capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))


def test_the_dry_run_needs_no_token_no_key_and_no_network(tmp_path):
    out = run_driver(tmp_path, "dry-run", "--ssh-pubkey", str(tmp_path / "missing.pub"))
    assert out.returncode == 0, out.stderr
    assert "<ssh-key-fingerprint>" in out.stdout and "<droplet-ip>" in out.stdout


def test_the_dry_run_prints_the_cost_table_first_then_every_command_in_order(tmp_path):
    out = run_driver(tmp_path, "dry-run").stdout
    assert out.index("## costed plan") < out.index("## the whole session")
    markers = ["stage.py", "driver.py preflight", "deadman.py", "driver.py create", "scp -r",
               "entrypoint.sh", "smoke.sh", "serve.sh --probe", "tunnel", "amd1-probe",
               "driver.py project", "run_sft.sh --arm A", "run_sft.sh --arm B", "run_sft.sh --arm AB",
               "serve.sh --all", "amd1-base --agent bash --rungs L1", "--rungs L2", "--rungs L3",
               "--rungs L4", "amd1-base-program", "sync_back.sh", "verify-sync", "driver.py destroy"]
    pos = [out.index(m) for m in markers]
    assert pos == sorted(pos), dict(zip(markers, pos))


def test_the_dry_run_names_region_size_image_fingerprint_and_tag_in_the_create_request(tmp_path):
    out = run_driver(tmp_path, "dry-run", "--ssh-key-fingerprint", "de:ad:be:ef").stdout
    create = next(l for l in out.splitlines() if l.startswith("#   POST https"))
    body = json.loads(create.split("  ", 2)[2].strip())
    assert body["region"] == "ric1" and body["size"] == "gpu-mi350x1-288gb-spot"
    assert body["image"] == "amddeveloperclou-vllm0171" and body["ssh_keys"] == ["de:ad:be:ef"]
    assert body["tags"] == ["smol-ladder"]


def test_the_dry_run_ends_with_destroy_and_the_follow_up_get(tmp_path):
    out = run_driver(tmp_path, "dry-run").stdout
    tail = out[out.rindex("[destroy] destroy"):]
    assert "GET /v2/droplets?tag_name=smol-ladder" in tail and "must be empty" in tail


def test_the_dry_run_leaves_no_ledger_behind(tmp_path):
    run_driver(tmp_path, "dry-run")
    assert not (tmp_path / "l.jsonl").exists()


def test_plan_json_is_the_cost_table(tmp_path):
    out = run_driver(tmp_path, "plan", "--json").stdout
    table = json.loads(out[out.index("{"):])
    assert table["price_per_hour"] == 2.46 and table["within_budget"] is True
    assert table["total_dollars"] == pytest.approx(sum(r["dollars"] for r in table["rows"]), abs=0.1)


def test_the_fallback_flag_switches_size_region_price_together(tmp_path):
    out = run_driver(tmp_path, "dry-run", "--fallback").stdout
    assert "gpu-mi325x1-256gb" in out and '"region": "tor1"' in out and "$3.8/h" in out


def test_status_works_with_an_empty_ledger_and_no_token(tmp_path):
    out = run_driver(tmp_path, "status")
    assert out.returncode == 0 and "nothing should be billing" in out.stdout


def test_create_without_yes_prints_the_request_and_sends_nothing(tmp_path):
    out = run_driver(tmp_path, "create", "--ssh-key-fingerprint", "aa:bb")
    assert out.returncode == 0 and "not sent" in out.stdout
    assert not (tmp_path / "l.jsonl").exists()      # nothing recorded: nothing started billing


def test_destroy_without_yes_prints_and_sends_nothing(tmp_path):
    out = run_driver(tmp_path, "destroy")
    assert out.returncode == 0 and "not sent" in out.stdout


def test_train_and_go_refuse_without_a_go_on_record(tmp_path):
    for cmd in ("train", "go"):
        out = run_driver(tmp_path, cmd)
        assert out.returncode != 0 and "no GO on record" in out.stderr


def test_an_unknown_arm_is_refused_before_anything_runs(tmp_path):
    out = run_driver(tmp_path, "dry-run", "--arms", "A,Z")
    assert out.returncode != 0 and "unknown arm" in out.stderr


def test_note_prior_spend_lands_in_the_ledger_and_the_total(tmp_path):
    run_driver(tmp_path, "note-prior-spend", "12.5")
    assert L.spend(L.read(tmp_path / "l.jsonl"), T0).total == pytest.approx(12.5)


def test_no_planned_command_carries_a_secret_in_its_argv():
    flat_args = " ".join(a for s in plan_for() for c in s.cmds for a in c.argv)
    for needle in ("hf_", "HF_TOKEN", "DIGITALOCEAN_ACCESS_TOKEN", "AMD_CLOUD_API_TOKEN", "dop_v1"):
        assert needle not in flat_args


def test_the_remote_commands_use_exactly_the_scripts_that_exist():
    scripts = {a.split("/")[-1] for s in plan_for() if s.where == "droplet"
               for a in s.cmds[0].argv if a.endswith(".sh")}
    assert scripts == {"entrypoint.sh", "smoke.sh", "serve.sh", "run_sft.sh", "sync_back.sh"}
    for name in scripts:
        assert (OPS / name).exists()


# ═══ probe_tools: the tool-call classifier ══════════════════════════════════════════

def test_a_parsed_bash_tool_call_is_a_tool_call():
    msg = {"content": "", "tool_calls": [{"function": {"name": "bash",
                                                       "arguments": '{"command": "ls"}'}}]}
    assert probe_tools.classify(msg) == "tool_call"


def test_raw_tool_call_text_in_the_content_is_a_parser_mismatch_not_a_tool_call():
    assert probe_tools.classify({"content": "<tool_call>{...}</tool_call>"}) == "leaked"


def test_prose_and_empty_answers_are_not_tool_calls():
    assert probe_tools.classify({"content": "There are 3 rows."}) == "prose"
    assert probe_tools.classify({"content": ""}) == "empty"
    bad_args = {"content": "", "tool_calls": [{"function": {"name": "bash", "arguments": "{oops"}}]}
    assert probe_tools.classify(bad_args) == "empty"


# ═══ the shell scripts ══════════════════════════════════════════════════════════════

SCRIPTS = sorted(OPS.glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(script):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_every_script_is_shellcheck_clean():
    out = subprocess.run(["shellcheck", "-x", *map(str, SCRIPTS)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_no_script_runs_the_train_extra_which_would_replace_the_images_torch():
    for script in SCRIPTS:
        for line in script.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert ".[train]" not in line and "extra train" not in line, (script.name, line)


def test_the_entrypoint_installs_the_training_stack_under_a_torch_constraint():
    text = (OPS / "entrypoint.sh").read_text()
    assert 'torch==%s' in text and "-c " in text and "pip install" in text


def test_the_server_is_never_bound_to_a_public_interface():
    assert "--host 127.0.0.1" in (OPS / "serve.sh").read_text()
    assert "0.0.0.0" not in " ".join(l for l in (OPS / "serve.sh").read_text().splitlines()
                                     if not l.lstrip().startswith("#"))


def test_the_hub_repos_are_created_private_before_training():
    text = (OPS / "entrypoint.sh").read_text()
    assert "private=True" in text and "assert api.model_info" in text


def watchdog_env(tmp_path):
    return {"PATH": os.environ["PATH"], "AMD_REMOTE_ROOT": str(tmp_path / "root"),
            "AMD_REMOTE_LOG": str(tmp_path / "log"), "HOME": str(tmp_path)}


def test_the_watchdog_trips_on_the_wall_clock_and_names_the_reason(tmp_path):
    out = subprocess.run(["bash", str(OPS / "watchdog.sh"), "--once", "--dry-run",
                          "--max-minutes", "0", "--state-dir", str(tmp_path / "st")],
                         capture_output=True, text=True, env=watchdog_env(tmp_path))
    assert out.returncode == 1 and "FIRING: wall clock" in out.stderr
    assert "would push to the Hub and power off" in out.stderr


def test_the_watchdog_does_not_trip_when_every_limit_is_far_away(tmp_path):
    out = subprocess.run(["bash", str(OPS / "watchdog.sh"), "--once", "--dry-run",
                          "--max-minutes", "9999", "--idle-minutes", "9999", "--ssh-minutes", "9999",
                          "--state-dir", str(tmp_path / "st")],
                         capture_output=True, text=True, env=watchdog_env(tmp_path))
    assert out.returncode == 0 and "no trip" in out.stderr


# ═══ the runbook cannot drift from the code ═════════════════════════════════════════

def runbook() -> str:
    return RUNBOOK.read_text()


def test_the_runbook_quotes_the_total_the_driver_computes():
    text = runbook()
    m = re.search(r"Costed total at the planning placeholders: \*\*\$([\d.]+)\*\*", text)
    assert m, "the runbook has no parseable costed-total line"
    rows = P.projection(P.Config(), tokens_from_data(), P.Measured())
    assert float(m.group(1)) == pytest.approx(P.total_dollars(rows, P.PRICE_MI350X), abs=0.06)


def tokens_from_data() -> dict[str, P.SetTokens]:
    """The same token counts the runbook quotes: from tokens.json if staged, else the numbers
    measured with the Qwen3.5 tokenizer and recorded in the runbook itself."""
    text = runbook()
    a = int(re.search(r"A ([\d,]+) trained tokens", text).group(1).replace(",", ""))
    b = int(re.search(r"B ([\d,]+) trained tokens", text).group(1).replace(",", ""))
    return {"A": P.SetTokens(4439, a, "runbook"), "B": P.SetTokens(2029, b, "runbook"),
            "AB": P.SetTokens(6468, a + b, "runbook")}


def test_the_runbook_has_the_hardware_and_price_table():
    text = runbook()
    for needle in ("gpu-mi350x1-288gb-spot", "$2.46", "ric1", "gpu-mi325x1-256gb", "$3.80",
                   "tor1", "nyc2", "gpu-mi300x1-192gb", "$2.59", "gpu-mi355x1-288gb-spot", "$2.97",
                   "mem1"):
        assert needle in text, needle
    assert re.search(r"MI300X.{0,80}(no region|not offered|NO region)", text, re.I | re.S)


def test_the_runbook_records_the_credit_and_its_expiry():
    text = runbook()
    assert "2026-10-18" in text and "$100" in text and "$90" in text and "$35" in text


def test_the_runbook_lists_the_reviewer_commands_in_the_order_the_plan_runs_them():
    text = runbook()
    cmds = ["stage.py", "driver.py preflight", "deadman.py", "driver.py create", "driver.py bootstrap",
            "driver.py smoke", "driver.py train", "driver.py serve", "driver.py tunnel",
            "driver.py eval", "driver.py sync", "driver.py destroy"]
    pos = [text.index(c) for c in cmds]
    assert pos == sorted(pos), dict(zip(cmds, pos))
    assert "driver.py go" in text and "driver.py status" in text


def test_the_runbook_has_the_failure_playbook_the_brief_asks_for():
    text = runbook().lower()
    for needle in ("spot reclaim", "rocm", "fallback hardware", "vllm too old", "tool calls empty",
                   "lora", "remaining unknowns"):
        assert needle in text, needle


def test_the_runbook_states_the_protocol_flag_and_that_the_base_uses_it_too():
    text = runbook()
    assert "--agent bash" in text and "--agent program" in text
    assert "SmolDataEnvs-sft" in text


def test_the_runbook_no_longer_plans_on_the_old_hardware_or_an_invented_wheel_index():
    text = runbook()
    assert "wheels.vllm.ai" not in text
    assert "bootstrap.sh" not in text and "run_eval.sh" not in text   # replaced by entrypoint.sh / laptop eval


# ═══ the $95 hard cutoff ════════════════════════════════════════════════════════════

def test_the_hard_limit_is_95_and_the_working_defaults_are_unchanged():
    assert P.HARD_TOTAL_LIMIT == 95.0
    assert (P.TOTAL_CAP, P.DEFAULT_BUDGET) == (90.0, 35.0)
    assert P.effective_total_cap(1000.0) == 95.0 and P.effective_total_cap(40.0) == 40.0


def test_a_config_above_the_hard_limit_cannot_exist():
    with pytest.raises(ValueError, match="hard limit"):
        P.Config(total_cap=95.01)
    assert P.Config(total_cap=95.0).total_cap == 95.0


def test_argparse_rejects_a_total_cap_above_95_for_the_driver_and_the_deadman(tmp_path):
    out = run_driver(tmp_path, "plan", "--total-cap", "96")
    assert out.returncode == 2 and "HARD total limit" in out.stderr
    ok = run_driver(tmp_path, "plan", "--total-cap", "95")
    assert ok.returncode == 0, ok.stderr
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "AMD_OFFLINE": "1"}
    dm = subprocess.run([sys.executable, str(OPS / "deadman.py"), "--deadline-minutes", "5",
                         "--total-cap", "120"], capture_output=True, text=True, env=env,
                        cwd=str(REPO_ROOT))
    assert dm.returncode == 2 and "HARD total limit" in dm.stderr


def test_no_environment_variable_can_raise_the_cap(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "AMD_OFFLINE": "1",
           "AMD_TOTAL_CAP": "500", "TOTAL_CAP": "500", "AMD_HARD_TOTAL_LIMIT": "500"}
    out = subprocess.run([sys.executable, str(OPS / "driver.py"), "plan", "--ledger",
                          str(tmp_path / "l.jsonl")], capture_output=True, text=True, env=env,
                         cwd=str(REPO_ROOT))
    assert "total cap $90.00 (HARD limit $95.00)" in out.stdout


def test_the_deadman_destroys_before_the_hard_limit_even_with_a_huge_cap_passed_in():
    # total_cap=1000 is what a buggy caller might pass; the limit is 95 - 0.50 for the destroy.
    below = deadman.decide(T0, T0 + 99 * HOUR, 1.0, 94.49, 1e6, 1000.0, [droplet()])
    at = deadman.decide(T0, T0 + 99 * HOUR, 1.0, 94.50, 1e6, 1000.0, [droplet()])
    assert not below.destroy and at.destroy and "HARD limit" in at.reason


def test_the_deadman_uses_the_lower_of_the_working_cap_and_the_hard_limit():
    d = deadman.decide(T0, T0 + 99 * HOUR, 1.0, 89.5, 35.0, 90.0, [droplet()])
    assert d.destroy and "total cap" in d.reason


def test_the_gate_refuses_any_step_that_would_pass_95_whatever_the_cap_says():
    prior = [{"event": L.PRIOR, "ts": T0, "dollars": 94.0}]
    v = L.verdict(prior, T0, HOUR, 2.46, 1e6, 1e6)      # 94 + 2.46 > 95
    assert not v.allowed and "HARD total limit" in v.reason
    assert L.verdict(prior, T0, 0.3 * HOUR, 2.46, 1e6, 1e6).allowed


def test_the_driver_gate_stops_a_step_at_the_hard_limit(tmp_path):
    led = ledger_with(tmp_path, (L.PRIOR, T0, {"dollars": 94.5}))
    c = cfg(ledger=str(led), budget=1e6, total_cap=95.0)
    with pytest.raises(SystemExit) as exc:
        driver.gate(c, P.Step("train", "sft-A", "droplet", seconds=HOUR), now=T0)
    assert "HARD total limit" in str(exc.value)


# ═══ credentials and the API client's error handling ════════════════════════════════

from ops.amd import doapi  # noqa: E402


def test_the_repo_specific_token_name_wins_over_the_generic_one(monkeypatch):
    monkeypatch.setenv("DIGITALOCEAN_ACCESS_TOKEN", "generic")
    monkeypatch.setenv("AMD_CLOUD_API_TOKEN", "specific")
    assert doapi.token_from_env() == "specific"
    monkeypatch.delenv("AMD_CLOUD_API_TOKEN")
    assert doapi.token_from_env() == "generic"


def test_dotenv_wins_over_a_stale_shell_variable_and_drops_the_other_token_name(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport AMD_CLOUD_API_TOKEN='fresh'\nHF_TOKEN=hf_new\n")
    monkeypatch.setenv("AMD_CLOUD_API_TOKEN", "stale")
    monkeypatch.setenv("HF_TOKEN", "hf_old")
    monkeypatch.setenv("DIGITALOCEAN_ACCESS_TOKEN", "stale-generic")
    doapi.load_dotenv(env)
    assert os.environ["AMD_CLOUD_API_TOKEN"] == "fresh" and os.environ["HF_TOKEN"] == "hf_new"
    assert "DIGITALOCEAN_ACCESS_TOKEN" not in os.environ
    assert doapi.token_from_env() == "fresh"


def test_dotenv_leaves_the_shell_token_alone_when_the_file_names_none(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("AMD_HUB_NAMESPACE=me\n")
    monkeypatch.setenv("DIGITALOCEAN_ACCESS_TOKEN", "shell")
    doapi.load_dotenv(tmp_path / ".env")
    assert doapi.token_from_env() == "shell"


def test_child_processes_never_inherit_a_token(monkeypatch):
    for name in doapi.SECRET_VARS:
        monkeypatch.setenv(name, "SECRET")
    env = doapi.child_env({"SMOL_LADDER_BASE_URL": "http://127.0.0.1:8000/v1"})
    assert not set(doapi.SECRET_VARS) & set(env)
    assert env["SMOL_LADDER_BASE_URL"].startswith("http://127.0.0.1") and "PATH" in env


def test_every_subprocess_the_driver_starts_gets_the_stripped_environment(monkeypatch):
    for name in doapi.SECRET_VARS:
        monkeypatch.setenv(name, "SECRET")
    seen = []

    class Spy:
        def __init__(self, *a, **kw):
            seen.append(kw.get("env")); self.returncode = 0; self.stdout = iter(())
        def wait(self, *a, **kw): return 0
        def poll(self): return 0
        def kill(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(driver.subprocess, "Popen", Spy)
    monkeypatch.setattr(driver.subprocess, "call", lambda *a, **kw: seen.append(kw.get("env")) or 0)
    c = cfg()
    driver.run_cmd(P.ssh(c, ["true"]), c.host)
    driver.run_cmd(P.ssh(c, ["true"]), c.host, capture=True)
    driver.run_parallel([P.ssh(c, ["true"]), P.ssh(c, ["true"])], c.host)
    driver.ensure_tunnel(c)
    assert seen and all(e is not None and not set(doapi.SECRET_VARS) & set(e) for e in seen), seen


class _Resp:
    def __init__(self, status=200, raw=b"{}", read_exc=None):
        self.status, self._raw, self._exc = status, raw, read_exc
    def read(self):
        if self._exc:
            raise self._exc
        return self._raw
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.mark.parametrize("resp", [
    _Resp(read_exc=TimeoutError("read timed out")),
    _Resp(read_exc=__import__("http.client").client.IncompleteRead(b"{")),
    _Resp(raw=b"<html>502 bad gateway</html>"),
    _Resp(raw=b"[1, 2]"),
], ids=["read-timeout", "incomplete-read", "non-json-200", "json-but-not-an-object"])
def test_a_read_or_parse_failure_is_status_zero_not_an_exception_and_not_an_empty_listing(
        monkeypatch, resp):
    monkeypatch.setattr(doapi.urllib.request, "urlopen", lambda *a, **kw: resp)
    status, body = DoApi("t").get("/droplets")
    assert status == 0 and "error" in body
    # ... which the listing helper turns into a failure, never into "no droplets"
    class Api:
        def get(self, path): return DoApi("t").get(path)
    with pytest.raises(SystemExit):
        cloud.tagged(Api(), "smol-ladder")


def test_a_200_without_a_droplets_list_is_a_failed_listing():
    class Api:
        def get(self, path): return 200, {"meta": {}}
    with pytest.raises(SystemExit):
        cloud.tagged(Api(), "smol-ladder")


# ═══ the deadman loop, its heartbeat, and the driver's gate on it ═══════════════════

class Clock:
    def __init__(self, t=T0):
        self.t = t
    def __call__(self):
        return self.t
    def sleep(self, s):
        self.t += s


class _Stop(BaseException):
    """Raised by a test's sleep to end a loop that, by design, would never return."""


class LoopApi:
    """Scripted tag listings: `script` is consumed one GET at a time; an Exception is raised, a
    (status, body) tuple returned, and a list becomes the droplets. Past the end: the last item."""

    def __init__(self, script, delete_clears_after=None):
        self.script, self.calls, self.deletes = list(script), [], 0
        self.delete_clears_after = delete_clears_after
        self.last = None

    def get(self, path):
        self.calls.append(path)
        if path.startswith("/droplets/"):
            return (404, {}) if self.deletes and (self.delete_clears_after is not None
                                                  and self.deletes >= self.delete_clears_after) \
                else (200, {"droplet": {"id": 1}})
        if path.startswith("/droplets?per_page"):
            return 200, {"droplets": []}
        item = self.script.pop(0) if self.script else self.last
        self.last = item
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return item
        if self.delete_clears_after is not None and self.deletes >= self.delete_clears_after:
            return 200, {"droplets": []}
        return 200, {"droplets": item}

    def delete(self, path):
        self.deletes += 1
        return 204, {}


def watch_args(tmp_path, clock, **over):
    kw = dict(tag="smol-ladder", ledger_path=tmp_path / "l.jsonl", deadline=T0 + HOUR,
              session_cap=35.0, total_cap=90.0, price=2.46, poll=30.0, sleep=clock.sleep,
              clock=clock, out=lambda s: None)
    kw.update(over)
    return kw


def test_the_deadman_loop_survives_api_exceptions_and_keeps_ticking(tmp_path):
    clock, lines = Clock(), []
    api = LoopApi([TimeoutError("read timed out"), __import__("http.client").client.IncompleteRead(b"x"),
                   ValueError("not json"), (500, {"error": "boom"}), []])
    # deadline long past: once a listing succeeds and shows nothing, it exits cleanly.
    code = deadman.watch(api, **watch_args(tmp_path, clock, deadline=T0 - 2 * HOUR, out=lines.append))
    assert code == 0
    assert sum("tick failed" in l for l in lines) == 4 and any("exiting" in l for l in lines)
    assert len([c for c in api.calls if c.startswith("/droplets?tag_name")]) == 5


def test_the_backoff_grows_between_failures_and_is_capped(tmp_path):
    clock, waits = Clock(), []
    def sleep(s):
        waits.append(s)
        if len(waits) == 9:
            raise _Stop
        clock.sleep(s)
    with pytest.raises(_Stop):
        deadman.watch(LoopApi([OSError("down")]), **watch_args(tmp_path, clock, sleep=sleep))
    assert waits[0] < waits[1] < waits[2] and max(waits) == deadman.MAX_BACKOFF_SECONDS


def test_the_deadman_refuses_to_exit_while_a_destroy_is_unverified(tmp_path):
    clock, sleeps, lines = Clock(), [0], []
    def sleep(s):
        sleeps[0] += 1
        if sleeps[0] > 60:
            raise _Stop
        clock.sleep(s)
    api = LoopApi([[droplet()]])               # the droplet never goes away
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    with pytest.raises(_Stop):                  # it never returned: it was still trying when stopped
        deadman.watch(api, **watch_args(tmp_path, clock, deadline=T0 + 10, sleep=sleep, out=lines.append))
    assert api.deletes >= 3 and any("NOT verified" in l and "NOT exiting" in l for l in lines)
    assert L.open_interval(L.read(led)) is not None


def test_the_deadman_exits_zero_only_once_the_destroy_is_verified(tmp_path):
    clock, lines = Clock(), []
    api = LoopApi([[droplet()]], delete_clears_after=3)    # gone after the third DELETE
    ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    code = deadman.watch(api, **watch_args(tmp_path, clock, deadline=T0 + 10, out=lines.append))
    assert code == 0 and api.deletes == 3 and any("destroy verified" in l for l in lines)
    assert L.open_interval(L.read(tmp_path / "l.jsonl")) is None


def test_once_exits_nonzero_when_the_destroy_could_not_be_verified(tmp_path):
    clock = Clock()
    ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    assert deadman.watch(LoopApi([[droplet()]]), **watch_args(tmp_path, clock, deadline=T0 - 10,
                                                              once=True)) == 1


def test_the_destroys_ledger_events_carry_the_real_clock_not_the_decisions(tmp_path):
    clock = Clock(T0 + 2 * HOUR)
    api = LoopApi([[droplet()]], delete_clears_after=1)
    api_delete = api.delete
    api.delete = lambda path: (clock.sleep(40), api_delete(path))[1]     # the DELETE takes a while
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": 1}))
    decided_at = clock()
    d = deadman.tick(api, "smol-ladder", led, decided_at, T0 + HOUR, 35.0, 90.0, 2.46,
                     dry_run=False, sleep=clock.sleep, clock=clock)
    assert d.destroy and d.verified_gone
    destroyed = [e for e in L.read(led) if e["event"] == L.DESTROYED][-1]
    assert destroyed["ts"] > decided_at


def test_the_heartbeat_is_written_every_tick_and_records_the_last_successful_listing(tmp_path):
    clock = Clock()
    seen = []
    def sleep(s):
        seen.append(L.heartbeat_read(tmp_path / "l.jsonl"))
        if len(seen) == 3:
            raise _Stop
        clock.sleep(s)
    api = LoopApi([[], OSError("x"), []])
    with pytest.raises(_Stop):
        deadman.watch(api, **watch_args(tmp_path, clock, sleep=sleep, deadline=T0 + 9 * HOUR))
    assert seen[0]["state"] == "ok" and seen[0]["last_ok"] == T0
    assert seen[1]["state"] == "error" and seen[1]["last_ok"] == T0     # blind, and says so
    assert seen[0]["tag"] == "smol-ladder" and seen[0]["total_cap"] == 90.0


def hb(tmp_path, now, **over):
    kw = dict(last_ok=now, poll_seconds=30.0, tag="smol-ladder", deadline=now + HOUR, budget=35.0,
              total_cap=90.0)
    kw.update(over)
    L.heartbeat_write(tmp_path / "l.jsonl", now, **kw)
    return tmp_path / "l.jsonl"


def test_a_heartbeat_is_fresh_within_three_poll_intervals_and_stale_after(tmp_path):
    led = hb(tmp_path, T0)
    assert L.heartbeat_status(led, T0 + 89)[0] and not L.heartbeat_status(led, T0 + 91)[0]
    assert "stopped" in L.heartbeat_status(led, T0 + 500)[1]


def test_a_blind_or_mismatched_or_loose_deadman_is_not_a_live_one(tmp_path):
    assert not L.heartbeat_status(hb(tmp_path, T0, last_ok=T0 - 500), T0)[0]
    assert not L.heartbeat_status(hb(tmp_path, T0), T0, tag="other")[0]
    assert not L.heartbeat_status(hb(tmp_path, T0, budget=50.0), T0, budget=35.0)[0]
    assert not L.heartbeat_status(hb(tmp_path, T0, total_cap=95.0), T0, total_cap=90.0)[0]
    assert not L.heartbeat_status(hb(tmp_path, T0, deadline=T0 - 1), T0)[0]
    assert not L.heartbeat_status(tmp_path / "missing.jsonl", T0)[0]
    (tmp_path / "bad.jsonl.heartbeat").write_text("{not json")
    assert not L.heartbeat_status(tmp_path / "bad.jsonl", T0)[0]


def test_the_gate_refuses_create_and_every_billed_step_without_a_fresh_heartbeat(tmp_path):
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    steps = {s.name: s for s in plan_for(c)}
    for name in ("create", "bootstrap", "smoke-checks", "sft-A", "serve"):
        with pytest.raises(SystemExit) as exc:
            driver.gate(c, steps[name], now=T0)
        msg = str(exc.value)
        assert f"STOPPING BEFORE '{name}'" in msg and "setsid nohup python ops/amd/deadman.py" in msg
        assert "re-arm" in msg


def test_a_fresh_heartbeat_lets_the_steps_through(tmp_path):
    c = cfg(ledger=str(hb(tmp_path, T0)))
    for name in ("create", "bootstrap", "sft-A"):
        driver.gate(c, next(s for s in plan_for(c) if s.name == name), now=T0 + 10)


def test_sync_and_destroy_are_never_blocked_by_a_dead_deadman(tmp_path, capsys):
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    for name in ("sync-droplet", "sync-pull", "destroy"):
        driver.gate(c, next(s for s in plan_for(c) if s.name == name), now=T0)
    assert "runs anyway" in capsys.readouterr().out


def test_run_steps_will_not_post_a_create_without_a_deadman(tmp_path):
    c = cfg(ledger=str(tmp_path / "l.jsonl"), local_logs=str(tmp_path / "logs"))
    api = FakeApi()
    with pytest.raises(SystemExit) as exc:
        driver.run_steps(c, plan_for(c), api, only="create")
    assert "No live dead-man switch" in str(exc.value) and mutations(api) == []


def test_a_deadman_with_no_token_exits_loudly_and_writes_no_heartbeat(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "AMD_OFFLINE": "1"}
    out = subprocess.run([sys.executable, str(OPS / "deadman.py"), "--deadline-minutes", "5",
                          "--ledger", str(tmp_path / "l.jsonl")], capture_output=True, text=True,
                         env=env, cwd=str(REPO_ROOT))
    assert out.returncode != 0 and "NOT ARMED" in out.stderr
    assert not (tmp_path / "l.jsonl.heartbeat").exists()      # so the driver's gate refuses


def test_the_deadman_command_in_the_refusal_is_detached_and_priced_for_the_hardware(tmp_path):
    c = cfg(ledger=str(tmp_path / "l.jsonl"), size=P.SIZE_MI325X, price=P.PRICE_MI325X)
    cmd = driver.deadman_command(c)
    assert cmd.startswith("setsid nohup python ops/amd/deadman.py") and "--price 3.8" in cmd
    assert "< /dev/null &" in cmd and f"--ledger {c.ledger}" in cmd


# ═══ create: an answer that does not say whether the droplet exists ═════════════════

class AmbiguousApi(FakeApi):
    """POST answers `post_status` with no droplet; the tag listing shows one only after
    `appears_after` listings (None = never)."""

    def __init__(self, post_status, appears_after=None, **kw):
        super().__init__(**kw)
        self.post_status, self.appears_after, self.listings = post_status, appears_after, 0

    def post(self, path, body):
        self.calls.append(("POST", path, body))
        return self.post_status, {"error": "gateway timeout"}

    def get(self, path):
        if path.startswith("/droplets?tag_name"):
            self.calls.append(("GET", path, None))
            self.listings += 1
            if self.appears_after is not None and self.listings > self.appears_after:
                return 200, {"droplets": [{"id": 42, "name": "smol-ladder", "status": "new"}]}
            return 200, {"droplets": []}
        return super().get(path)


@pytest.mark.parametrize("status", [0, 429, 500, 502, 503, 504, 408])
def test_an_ambiguous_create_never_closes_the_ledger_interval(tmp_path, status):
    api, led = AmbiguousApi(status), tmp_path / "l.jsonl"
    with pytest.raises(SystemExit) as exc:
        cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0)
    assert "UNKNOWN" in str(exc.value) and "Do NOT create again" in str(exc.value)
    ev = L.read(led)
    assert L.open_interval(ev) is not None and not any(e["event"] in L.CLOSING for e in ev)
    assert L.spend(ev, T0 + HOUR).open                    # still counted by every guard


def test_an_ambiguous_create_polls_the_tag_listing_for_about_90_seconds(tmp_path):
    api, naps = AmbiguousApi(0), []
    with pytest.raises(SystemExit):
        cloud.create(api, cfg(), tmp_path / "l.jsonl", sleep=naps.append, now=lambda: T0)
    assert 85 <= sum(naps) <= 100 and api.listings >= 18


def test_an_ambiguous_create_that_landed_is_adopted_and_carried_on(tmp_path):
    api, led = AmbiguousApi(504, appears_after=3), tmp_path / "l.jsonl"
    info = cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0)
    assert info["droplet_id"] == 42 and info["ip"]
    created = [e for e in L.read(led) if e["event"] == L.CREATED]
    assert created[-1]["droplet_id"] == 42
    assert len([c for c in api.calls if c[0] == "POST"]) == 1        # it did not POST twice


def test_a_listing_that_fails_during_resolution_does_not_resolve_anything(tmp_path):
    class Flaky(AmbiguousApi):
        def get(self, path):
            if path.startswith("/droplets?tag_name") and self.listings >= 1:
                self.listings += 1
                return 500, {}
            return super().get(path)
    api, led = Flaky(0), tmp_path / "l.jsonl"
    with pytest.raises(SystemExit) as exc:
        cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0)
    assert "listing errors" in str(exc.value) and L.open_interval(L.read(led)) is not None


def test_a_re_create_is_refused_while_the_previous_one_is_unresolved(tmp_path):
    led = tmp_path / "l.jsonl"
    with pytest.raises(SystemExit):
        cloud.create(AmbiguousApi(0), cfg(), led, sleep=lambda s: None, now=lambda: T0)
    api = FakeApi()
    with pytest.raises(SystemExit) as exc:
        cloud.create(api, cfg(), led, sleep=lambda s: None, now=lambda: T0 + 200)
    assert "open interval" in str(exc.value) and mutations(api) == []


def test_never_two_droplets_create_refuses_when_one_is_already_tagged(tmp_path):
    api = FakeApi(tagged=[droplet()])
    with pytest.raises(SystemExit) as exc:
        cloud.create(api, cfg(), tmp_path / "l.jsonl", sleep=lambda s: None, now=lambda: T0)
    assert "Never two droplets" in str(exc.value) and mutations(api) == []
    assert not (tmp_path / "l.jsonl").exists()


def test_create_refuses_when_it_cannot_list_the_tag(tmp_path):
    class Blind(FakeApi):
        def get(self, path):
            return (0, {"error": "network"}) if path.startswith("/droplets?tag_name") else super().get(path)
    api = Blind()
    with pytest.raises(SystemExit):
        cloud.create(api, cfg(), tmp_path / "l.jsonl", sleep=lambda s: None, now=lambda: T0)
    assert mutations(api) == []


def test_a_pending_create_is_not_written_off_by_reconcile_for_ten_minutes(tmp_path):
    led = ledger_with(tmp_path, (L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": None,
                                                "pending": True}))
    assert cloud.reconcile([], led, T0 + 300) is False
    assert cloud.reconcile([], led, T0 + 601) is True


# ═══ destroy: verified three ways ═══════════════════════════════════════════════════

def recorded(tmp_path, *ids):
    events = [(L.CREATED, T0, {"price_per_hour": 2.46, "droplet_id": i}) for i in ids[:1]]
    return ledger_with(tmp_path, *events)


def test_destroy_gets_every_droplet_id_the_ledger_recorded_and_expects_404(tmp_path):
    api = FakeApi(tagged=[droplet()])
    led = recorded(tmp_path, 42)
    assert cloud.destroy(api, "smol-ladder", led, sleep=lambda s: None, now=lambda: T0 + HOUR)
    assert ("GET", "/droplets/42", None) in api.calls
    assert L.read(led)[-1]["checked_ids"] == [42]


def test_a_droplet_that_lost_its_tag_but_still_answers_200_is_deleted_by_id_and_not_verified_yet(tmp_path):
    class Untagged(FakeApi):
        """The tag listing is empty (the tag is gone) but id 42 still exists until deleted by id."""
        def get(self, path):
            if path.startswith("/droplets?tag_name"):
                self.calls.append(("GET", path, None)); return 200, {"droplets": []}
            if path == "/droplets/42":
                self.calls.append(("GET", path, None))
                return (404, {}) if ("DELETE", "/droplets/42", None) in self.calls else (200, {"droplet": {"id": 42}})
            return super().get(path)
    api, led = Untagged(), recorded(tmp_path, 42)
    assert cloud.destroy(api, "smol-ladder", led, sleep=lambda s: None, now=lambda: T0 + HOUR)
    assert ("DELETE", "/droplets/42", None) in api.calls


def test_destroy_is_unverified_while_a_recorded_id_still_answers(tmp_path):
    class Stuck(FakeApi):
        def get(self, path):
            if path.startswith("/droplets?tag_name"):
                return 200, {"droplets": []}
            if path == "/droplets/42":
                return 200, {"droplet": {"id": 42}}
            return super().get(path)
    led = recorded(tmp_path, 42)
    assert cloud.destroy(Stuck(), "smol-ladder", led, sleep=lambda s: None, checks=3,
                         now=lambda: T0) is False
    assert L.open_interval(L.read(led)) is not None


def test_a_5xx_on_the_id_check_is_not_a_verification(tmp_path):
    class Flaky(FakeApi):
        def get(self, path):
            return (503, {}) if path == "/droplets/42" else super().get(path)
    led = recorded(tmp_path, 42)
    assert cloud.destroy(Flaky(tagged=[]), "smol-ladder", led, sleep=lambda s: None, checks=2,
                         now=lambda: T0) is False


def test_an_untagged_gpu_droplet_that_is_not_ours_is_reported_loudly_and_never_deleted(tmp_path, capsys):
    api = FakeApi(tagged=[droplet()])
    api.account_extra = [{"id": 999, "name": "someone-elses", "size_slug": "gpu-mi300x1-192gb",
                          "status": "active", "tags": []},
                         {"id": 998, "name": "web", "size_slug": "s-1vcpu-1gb", "status": "active"}]
    led = recorded(tmp_path, 42)
    assert cloud.destroy(api, "smol-ladder", led, sleep=lambda s: None, now=lambda: T0 + HOUR)
    out = capsys.readouterr().out
    assert "UNTAGGED GPU DROPLET" in out and "999" in out and "NOT touched" in out
    assert [c for c in mutations(api)] == [("DELETE", "/droplets?tag_name=smol-ladder", None)]
    note = next(e for e in L.read(led) if "UNEXPECTED" in e.get("text", ""))
    assert {d["id"] for d in note["droplets"]} == {999, 998}
    assert L.read(led)[-1]["event"] == L.DESTROYED


def test_a_failed_account_listing_is_reported_but_does_not_undo_a_verified_destroy(tmp_path, capsys):
    class NoAudit(FakeApi):
        def get(self, path):
            return (500, {}) if path.startswith("/droplets?per_page") else super().get(path)
    led = recorded(tmp_path, 42)
    assert cloud.destroy(NoAudit(tagged=[droplet()]), "smol-ladder", led, sleep=lambda s: None,
                         now=lambda: T0 + HOUR)
    assert "could not audit" in capsys.readouterr().out


# ═══ `go`: the destroy happens ══════════════════════════════════════════════════════

class GoRecorder:
    """Replaces run_steps/best_effort_sync inside `go`. `fail` maps a phase to an exception."""

    def __init__(self, fail=None):
        self.log, self.fail = [], fail or {}

    def run_steps(self, cfg, steps, api=None, only="", new_session=False):
        names = [s.name for s in steps]
        if only:
            self.log.append(only)
            if only in self.fail:
                raise self.fail[only]
        else:
            self.log.extend(names)

    def best_effort_sync(self, cfg, steps, **kw):
        self.log.append("best-effort-sync")
        return True


@pytest.fixture
def go_env(monkeypatch, tmp_path):
    rec = GoRecorder()
    monkeypatch.setattr(driver, "run_steps", rec.run_steps)
    monkeypatch.setattr(driver, "best_effort_sync", rec.best_effort_sync)
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    return rec, c, plan_for(c)


def test_go_runs_the_phases_in_order_then_tears_down_exactly_once(go_env):
    rec, c, steps = go_env
    assert driver.go(c, steps, None) == 0
    assert rec.log == ["train", "serve", "tunnel", "eval", "sync", "tunnel-down", "destroy"]


def test_go_reaches_destroy_when_a_step_raises_and_syncs_first(go_env):
    rec, c, steps = go_env
    rec.fail["eval"] = RuntimeError("the harness fell over")
    with pytest.raises(RuntimeError):
        driver.go(c, steps, None)
    assert rec.log == ["train", "serve", "tunnel", "eval", "best-effort-sync", "tunnel-down", "destroy"]


def test_go_reaches_destroy_when_a_gate_refuses_mid_run(go_env):
    rec, c, steps = go_env
    rec.fail["train"] = SystemExit("STOPPING BEFORE 'sft-B'")
    with pytest.raises(SystemExit):
        driver.go(c, steps, None)
    assert rec.log[-3:] == ["best-effort-sync", "tunnel-down", "destroy"]


def test_go_reaches_destroy_on_ctrl_c(go_env):
    rec, c, steps = go_env
    rec.fail["serve"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        driver.go(c, steps, None)
    assert rec.log[-1] == "destroy"


def test_a_failed_verify_sync_stops_before_the_destroy_and_says_so(go_env, capsys):
    rec, c, steps = go_env
    rec.fail["sync"] = driver.SyncUnverified("verify-sync FAILED")
    with pytest.raises(driver.SyncUnverified):
        driver.go(c, steps, None, now=lambda: T0)
    assert "destroy" not in rec.log and "best-effort-sync" not in rec.log
    out = capsys.readouterr().out
    assert "STOPPING BEFORE THE DESTROY" in out and "still up and BILLING" in out


def test_a_failed_verify_sync_is_overridden_when_the_budget_forces_the_destroy(go_env, capsys):
    rec, c, steps = go_env
    L.append(c.ledger, L.CREATED, now=T0, price_per_hour=2.46)
    rec.fail["sync"] = driver.SyncUnverified("verify-sync FAILED")
    with pytest.raises(driver.SyncUnverified):
        driver.go(c, steps, None, now=lambda: T0 + 14.2 * HOUR)   # 30 min more would pass $35
    assert rec.log[-3:] == ["best-effort-sync", "tunnel-down", "destroy"]
    out = capsys.readouterr().out
    assert "FORCED DESTROY" in out and "LOST" in out and "$35.00" in out


def test_a_failed_verify_sync_is_overridden_when_the_deadman_deadline_is_near(go_env):
    rec, c, steps = go_env
    L.heartbeat_write(c.ledger, T0, last_ok=T0, poll_seconds=30, tag="smol-ladder",
                      deadline=T0 + 600, budget=35.0, total_cap=90.0)
    rec.fail["sync"] = driver.SyncUnverified("verify-sync FAILED")
    with pytest.raises(driver.SyncUnverified):
        driver.go(c, steps, None, now=lambda: T0)
    assert rec.log[-1] == "destroy"


def test_best_effort_sync_is_bounded_guarded_and_never_blocks_the_destroy(tmp_path):
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    calls = []

    def runner(cmd, host, timeout=None):
        calls.append((cmd.argv[0], timeout))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return 1, ""
    assert driver.best_effort_sync(c, plan_for(c), runner=runner, timeout=99.0) is False
    assert [x[1] for x in calls] == [99.0, 99.0] and len(calls) == 2     # both tried, both bounded


GO_HARNESS = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    from ops.amd import driver, plan as P
    record = Path({record!r})
    def note(s):
        with record.open("a") as fh:
            fh.write(s + "\\n")
    def fake_run_steps(cfg, steps, api=None, only="", new_session=False):
        if only == "train":
            print("READY", flush=True)
            time.sleep(60)                      # "training", until the signal arrives
        elif not only:
            note("+".join(s.name for s in steps))   # the teardown: tunnel-down + destroy
    driver.run_steps = fake_run_steps
    driver.best_effort_sync = lambda cfg, steps, **kw: note("best-effort-sync")
    driver.install_signal_handlers()
    cfg = P.Config(host="203.0.113.9", fingerprint="aa", ledger={ledger!r})
    driver.go(cfg, P.build_plan(cfg, {{}}, P.Measured()), None)
''')


def run_go_harness(tmp_path, sig, second_signal=False):
    record = tmp_path / "record.txt"
    script = tmp_path / "harness.py"
    script.write_text(GO_HARNESS.format(repo=str(REPO_ROOT), record=str(record),
                                        ledger=str(tmp_path / "l.jsonl")))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                            stderr=subprocess.PIPE)
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(sig)
    if second_signal:
        proc.send_signal(sig)
    proc.wait(timeout=30)
    return proc.returncode, record.read_text().splitlines() if record.exists() else []


def test_sigterm_during_go_still_reaches_the_destroy(tmp_path):
    code, record = run_go_harness(tmp_path, signal.SIGTERM)
    assert record == ["best-effort-sync", "tunnel-down+destroy"], record
    assert code == 128 + signal.SIGTERM


def test_sighup_during_go_still_reaches_the_destroy(tmp_path):
    code, record = run_go_harness(tmp_path, signal.SIGHUP)
    assert record[-1] == "tunnel-down+destroy" and code == 128 + signal.SIGHUP


def test_a_signal_delivered_twice_does_not_cancel_the_cleanup(tmp_path):
    code, record = run_go_harness(tmp_path, signal.SIGTERM, second_signal=True)
    assert record[-1] == "tunnel-down+destroy"


def test_without_the_handlers_sigterm_would_skip_the_finally(tmp_path):
    """The reviewer's finding, demonstrated: the same harness without install_signal_handlers()."""
    script = tmp_path / "bare.py"
    record = tmp_path / "bare.txt"
    script.write_text(textwrap.dedent(f"""
        import time
        try:
            print("READY", flush=True); time.sleep(60)
        finally:
            open({str(record)!r}, "w").write("finally ran")
    """))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    assert not record.exists()


def test_the_driver_main_installs_the_handlers():
    assert "install_signal_handlers()" in inspect_source(driver.main)


def inspect_source(fn) -> str:
    import inspect
    return inspect.getsource(fn)
