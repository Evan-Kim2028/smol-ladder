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
    """The recounted tokens at max_length 8192 (A) and for ja3_sft_v2 (B): the plan's own numbers."""
    return {"A": P.SetTokens(4439, 9_085_233, "test"), "B": P.SetTokens(1122, 3_123_047, "test"),
            "AB": P.SetTokens(5561, 12_208_280, "test")}


def measured(**over) -> P.Measured:
    m = P.Measured(tokens_per_s=9000.0, batch_size=8, grad_accum=1, trials_per_min=25.0,
                   tool_calls_ok=True, adapter_differs=True, gate_go=True, checks_ok=True,
                   resume_ok=True)
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
    assert "0.16.2" in (OPS / "container_setup.sh").read_text()


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


def test_an_unmeasured_plan_says_what_each_number_rests_on_and_that_this_droplet_has_not_measured_it(capsys):
    rows = P.projection(cfg(), tokens(), P.Measured())
    sft = next(r for r in rows if r.stage == "sft A")
    assert not sft.measured and "5,446 tok/s measured in session 1" in sft.basis
    ev = next(r for r in rows if r.stage == P.ROW_L1)
    assert not ev.measured and "22 trials/min" in ev.basis and "session 1" in ev.basis
    boot = next(r for r in rows if r.stage == "bootstrap")
    assert "12 min" in boot.basis and boot.seconds == 720.0


def test_a_measured_plan_marks_training_and_evaluation_rows_measured():
    rows = P.projection(cfg(), tokens(), measured())
    assert all(r.measured for r in rows if r.stage.startswith(("sft", "eval L1")))
    assert next(r for r in rows if r.stage == "sft A").basis.endswith("measured on this droplet")


def test_training_seconds_come_from_tokens_over_measured_throughput():
    rows = {r.stage: r for r in P.projection(cfg(), tokens(), measured(tokens_per_s=10_000.0))}
    assert rows["sft A"].seconds == pytest.approx(9_085_233 / 10_000.0 * P.SAFETY + P.SFT_OVERHEAD_S)
    assert rows["sft AB"].seconds > rows["sft A"].seconds + rows["sft B"].seconds - 400


def test_the_session_1_training_rate_costs_the_corrected_token_counts():
    rows = {r.stage: r for r in P.projection(cfg(), tokens(), P.Measured())}
    assert P.MEASURED_TOKENS_PER_S == 5446.0
    assert rows["sft A"].seconds == pytest.approx(9_085_233 / 5446.0 * P.SAFETY + P.SFT_OVERHEAD_S)
    assert rows["sft B"].seconds == pytest.approx(3_123_047 / 5446.0 * P.SAFETY + P.SFT_OVERHEAD_S)
    assert rows["sft AB"].seconds == pytest.approx(12_208_280 / 5446.0 * P.SAFETY + P.SFT_OVERHEAD_S)


def test_a_zero_throughput_is_refused_not_divided_by():
    with pytest.raises(ValueError):
        P.sft_seconds(1_000_000, 0.0)


def test_one_l1_block_of_five_models_is_computed_not_guessed():
    """5 x 250 = 1,250 trials at 22 per minute is about 57 minutes; the gate already did 2 x 60 of them."""
    c = cfg()
    ev = P.eval_breakdown(c, P.MEASURED_TRIALS_PER_MIN)
    assert ev["L1"] == pytest.approx((5 * 250 - 2 * 60) / 22.0 * 60.0)
    assert ev["sample2"] == pytest.approx(5 * 250 / 22.0 * 60.0)
    assert 5 * 250 / 22.0 == pytest.approx(56.8, abs=0.1)
    assert ev["gate"] == pytest.approx(2 * 60 / 22.0 * 60.0)
    assert ev["control"] == pytest.approx(250 * P.PROGRAM_COST_FACTOR / 22.0 * 60.0)
    assert ev["hints"] == pytest.approx(3 * 5 * 250 / 22.0 * 60.0)


def test_evaluation_time_scales_with_trials_and_inversely_with_the_rate():
    c = cfg(limit=40, rungs=("L1", "L2"))
    one = P.eval_breakdown(c, 20.0)
    assert one["hints"] == pytest.approx(5 * 40 / 20.0 * 60.0)
    half = P.eval_breakdown(c, 10.0)
    assert half["hints"] == pytest.approx(2 * one["hints"])
    assert P.eval_breakdown(cfg(limit=40, rungs=("L1",)), 20.0)["hints"] == 0.0


def test_a_slower_gate_lowers_the_rate_the_evaluation_is_costed_at_but_a_faster_one_does_not_raise_it():
    slow = P.trials_per_min_for(P.Measured(trials_per_min=11.0))
    assert slow[0] == 11.0 and "gate" in slow[1]
    fast = P.trials_per_min_for(P.Measured(trials_per_min=60.0))
    assert fast[0] == P.MEASURED_TRIALS_PER_MIN        # two models say nothing good about five


def test_the_default_plan_fits_the_session_budget_it_is_costed_against():
    rows = P.projection(cfg(), tokens(), P.Measured())
    st = P.staged_dollars(rows, P.PRICE_MI350X)
    assert st["core"] < P.DEFAULT_BUDGET
    assert st["all"] < P.TOTAL_CAP
    assert st["gate_only"] < st["core"] < st["all"]


def test_the_session_totals_are_gate_only_the_plan_and_each_optional_stage():
    rows = P.projection(cfg(), tokens(), P.Measured())
    price = P.PRICE_MI350X
    st = P.staged_dollars(rows, price)
    by = {r.stage: r.dollars(price) for r in rows}
    through_gate = sum(r.dollars(price) for r in rows[:[r.stage for r in rows].index(P.ROW_GATE_EVAL) + 1])
    assert st["gate_only"] == pytest.approx(through_gate + by["destroy + verify"])
    assert st["sample2"] == pytest.approx(by[P.ROW_SAMPLE2])
    assert st["hints"] == pytest.approx(by[P.ROW_HINTS]) and st["control"] == pytest.approx(by[P.ROW_CONTROL])
    assert st["core"] + st["sample2"] + st["hints"] + st["control"] == pytest.approx(st["all"])
    assert st["all"] == pytest.approx(P.total_dollars(rows, price))


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
    (tmp_path / "train" / "ja3_sft_v2.jsonl").write_text('{"messages": []}\n' * 10)
    t = P.heuristic_tokens(tmp_path, 8192)
    assert t["AB"].trained_tokens == t["A"].trained_tokens + t["B"].trained_tokens
    assert "pessimistic" in t["A"].method


def test_with_the_real_row_counts_the_fallback_is_the_corrected_recount_and_ab_is_a_plus_b(tmp_path):
    (tmp_path / "train" / "sft_upstream").mkdir(parents=True)
    (tmp_path / "train" / "sft_upstream" / "train.jsonl").write_text("{}\n" * 4439)
    (tmp_path / "train" / "ja3_sft_v2.jsonl").write_text("{}\n" * 1122)
    t = P.heuristic_tokens(tmp_path, 8192)
    assert (t["A"].trained_tokens, t["B"].trained_tokens) == (9_085_233, 3_123_047)
    assert t["AB"].trained_tokens == 12_208_280 and t["AB"].rows == 4439 + 1122


def test_staged_token_counts_from_other_data_are_refused(tmp_path):
    (tmp_path / "train" / "sft_upstream").mkdir(parents=True)
    (tmp_path / "train" / "sft_upstream" / "train.jsonl").write_text("{}\n" * 4439)
    (tmp_path / "train" / "ja3_sft_v2.jsonl").write_text("{}\n" * 1122)
    v1 = {"A": P.SetTokens(4439, 8_695_863, "x"), "B": P.SetTokens(2029, 6_203_984, "x"),
          "AB": P.SetTokens(6468, 14_899_847, "x")}
    assert "2029 rows for arm B" in P.stale_tokens(v1, tmp_path)
    assert P.stale_tokens(tokens(), tmp_path) == ""
    bad = {**tokens(), "AB": P.SetTokens(5561, 1, "x")}
    assert "AB is not A + B" in P.stale_tokens(bad, tmp_path)


def test_the_deadline_is_bounded_by_what_the_budget_buys():
    rows = [P.Row("x", 100 * HOUR, "", False)]
    assert P.default_deadline_minutes(cfg(), rows) == pytest.approx(35.0 / 2.46 * 60, abs=1)


# ═══ step ordering ══════════════════════════════════════════════════════════════════

def test_the_session_runs_in_the_order_the_reviewer_runs_it():
    n = names(plan_for())
    order = ["stage-inputs", "preflight", "deadman", "create", "wait-ssh", "clean-stage", "upload",
             "bootstrap", "smoke-checks", "smoke-pull", "gate-serve", "gate-tunnel", "gate-eval",
             "stop-gate-server", "gate-decide", "go-no-go",
             "sft-A", "sft-B", "sft-AB", "serve", "tunnel", "eval-L1", "eval-sample2", "eval-L2",
             "eval-L3", "eval-L4", "eval-control", "sync-droplet", "sync-pull", "verify-sync",
             "tunnel-down", "destroy"]
    assert n == order


def test_everything_that_can_happen_before_the_droplet_exists_does():
    steps = plan_for()
    created = names(steps).index("create")
    before = steps[:created]
    assert [s.name for s in before] == ["stage-inputs", "preflight", "deadman"]
    assert all(not s.billed for s in before)


def test_no_step_waits_on_a_human_and_the_stops_are_the_gate_and_the_go_no_go():
    steps = plan_for()
    assert not any("input(" in " ".join(c.argv) or "read -p" in " ".join(c.argv)
                   for s in steps for c in s.cmds)
    assert "STOPS unless the gate passed" in next(s for s in steps if s.name == "go-no-go").note
    assert "--accept-gate" in next(s for s in steps if s.name == "gate-decide").note


def test_the_gate_runs_before_any_arm_trains_and_the_go_no_go_needs_it():
    n = names(plan_for())
    for arm in ("sft-A", "sft-B", "sft-AB"):
        assert n.index("gate-decide") < n.index("go-no-go") < n.index(arm)
    assert n.index("gate-serve") < n.index("gate-tunnel") < n.index("gate-eval") < n.index("stop-gate-server")
    assert n.index("stop-gate-server") < n.index("gate-decide")     # the meter is not left running for a human
    stop = next(s for s in plan_for() if s.name == "stop-gate-server")
    assert stop.reserve is False and list(stop.cmds[0].argv[-1:]) == ["--stop"]


def test_the_arms_train_in_order_a_then_b_then_ab():
    n = names(plan_for())
    assert n.index("sft-A") < n.index("sft-B") < n.index("sft-AB") < n.index("serve")


def test_serving_happens_once_for_all_models_after_all_training():
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


def argv_of(cmd, flag):
    return cmd.argv[cmd.argv.index(flag) + 1]


def test_the_harness_runs_on_the_laptop_not_the_droplet():
    assert all(s.where == "laptop" for s in eval_steps())
    assert not any("run_ladder" in " ".join(c.argv) for s in plan_for() if s.where == "droplet"
                   for c in s.cmds)


def test_the_evaluations_point_each_model_at_its_own_port_with_the_non_thinking_template():
    arms = {"amd-base-2b": "base", "amd-a-2b": "A", "amd-b-2b": "B", "amd-ab-2b": "AB", "amd-r-2b": "R"}
    for s in eval_steps():
        for cmd in s.cmds:
            env = dict(cmd.env)
            arm = arms[argv_of(cmd, "--model")]
            assert env["SMOL_LADDER_BASE_URL"] == f"http://127.0.0.1:{P.port_for(cfg(), arm)}/v1"
            assert json.loads(env["SMOL_LADDER_CHAT_TEMPLATE_KWARGS"]) == {"enable_thinking": False}


def test_each_harness_process_has_its_own_scratch_root_through_the_env_override():
    cmds = eval_steps()[0].cmds
    scratch = [dict(c.env)["SMOL_LADDER_SCRATCH"] for c in cmds]
    assert len(set(scratch)) == len(cmds) == 5
    assert "SMOL_LADDER_SCRATCH" in (REPO_ROOT / "smol_ladder/run_ladder.py").read_text()


def test_the_protocol_is_the_upstream_bash_agent_for_every_model_and_the_stop_flag_is_the_default():
    """The SFT data is `FineEnvs/SmolDataEnvs-sft`: one tool named `bash`. In this repo that is
    `run_ladder --agent bash`. The base control uses the same protocol so that adapter minus base
    is the effect of SFT. `--bash-stop` is never passed: the harness default (model) is the
    policy the rows were made under, and passing `submit` would change what is measured."""
    for rung in ("L1", "L2", "L3", "L4"):
        step = next(s for s in eval_steps() if s.name == f"eval-{rung}")
        assert len(step.cmds) == 5
        for cmd in step.cmds:
            assert argv_of(cmd, "--agent") == "bash" and argv_of(cmd, "--max-turns") == "16"
            assert "--bash-stop" not in cmd.argv
    src = (REPO_ROOT / "smol_ladder/run_ladder.py").read_text()
    assert 'choices=["submit", "model"], default="model"' in src


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
    assert argv[argv.index("--run-tag") + 1] == "amd2-base-program"


def test_run_tags_are_fresh_for_this_session_so_session_1s_results_are_never_reused():
    assert P.Config().tag_prefix == "amd2" != "amd1"


def test_each_model_has_its_own_run_tag_and_the_model_name_the_server_exposes():
    step = next(s for s in eval_steps() if s.name == "eval-L1")
    got = {argv_of(c, "--run-tag"): argv_of(c, "--model") for c in step.cmds}
    assert got == {"amd2-base": "amd-base-2b", "amd2-a": "amd-a-2b", "amd2-b": "amd-b-2b",
                   "amd2-ab": "amd-ab-2b", "amd2-r": "amd-r-2b"}


def test_the_stages_are_separate_purchases_l1_first_one_sample_then_the_rest():
    n = [s.name for s in eval_steps()]
    assert n == ["eval-L1", "eval-sample2", "eval-L2", "eval-L3", "eval-L4", "eval-control"]
    steps = {s.name: s for s in eval_steps()}
    assert argv_of(steps["eval-L1"].cmds[0], "--samples") == "1"
    assert argv_of(steps["eval-sample2"].cmds[0], "--samples") == "2"
    assert argv_of(steps["eval-sample2"].cmds[0], "--run-tag") == argv_of(steps["eval-L1"].cmds[0], "--run-tag")
    assert argv_of(steps["eval-L1"].cmds[0], "--limit") == "250"
    assert argv_of(steps["eval-L2"].cmds[0], "--samples") == "1"


def test_flags_decide_samples_limit_workers_and_rungs():
    c = cfg(samples=3, late_samples=2, limit=17, rungs=("L1", "L2"), workers=5)
    steps = {s.name: s for s in eval_steps(c)}
    assert set(steps) == {"eval-L1", "eval-L2", "eval-control"}     # samples=3: no second-sample stage
    a1, a2 = steps["eval-L1"].cmds[0].argv, steps["eval-L2"].cmds[0].argv
    assert a1[a1.index("--samples") + 1] == "3" and a2[a2.index("--samples") + 1] == "2"
    assert a1[a1.index("--limit") + 1] == "17" and a1[a1.index("--workers") + 1] == "5"
    d = eval_steps()[0].cmds[0].argv
    assert d[d.index("--workers") + 1] == "8" and d[d.index("--limit") + 1] == "250"


def test_every_requested_rung_runs_on_every_task_so_denominators_are_equal():
    for s in eval_steps():
        assert "--no-climb" in s.cmds[0].argv


def test_dropping_the_base_removes_it_and_its_program_control():
    steps = eval_steps(cfg(include_base=False))
    assert "eval-control" not in [s.name for s in steps]
    assert all("amd-base-2b" not in c.shell() for s in steps for c in s.cmds)
    assert len(steps[0].cmds) == 4


def test_the_released_adapter_is_an_eval_only_arm_in_serve_tunnel_eval_verify_and_the_table():
    c = cfg()
    assert P.eval_only_arms(c) == ["R"] and P.served_name("R") == "amd-r-2b" and P.run_tag(c, "R") == "amd2-r"
    assert P.eval_arms(c) == ["base", "A", "B", "AB", "R"] and P.served_adapters(c) == ["A", "B", "AB", "R"]
    serve = next(s for s in plan_for(c) if s.name == "serve").cmds[0].argv
    assert "--hub" in serve and serve[serve.index("--hub") + 1] == "R=AdithyaSK/smoldataenvs-sft-2b-v0:8004"
    assert serve[serve.index("--arms") + 1] == "A,B,AB" and "--verify" in serve
    assert P.ports(c) == [8000, 8001, 8002, 8003, 8004]
    assert set(driver.required_trials(c)) == {"amd2-base", "amd2-a", "amd2-b", "amd2-ab", "amd2-r",
                                              "amd2-base-program"}
    rows = {r.stage: r for r in P.projection(c, tokens(), P.Measured())}
    assert "5 models" in rows[P.ROW_L1].basis
    # R is evaluated, never trained or pushed
    assert not any("sft-R" == s.name for s in plan_for(c))
    sync = next(s for s in plan_for(c) if s.name == "sync-droplet").cmds[0].argv
    assert sync[sync.index("--arms") + 1] == "A,B,AB"


def test_any_hub_adapter_can_be_added_and_it_gets_a_port_a_tag_and_a_tunnel_forward():
    c = cfg(eval_only=(("R", "AdithyaSK/smoldataenvs-sft-2b-v0"), ("X", "someone/else-sft")))
    assert P.port_for(c, "X") == 8005 and P.served_name("X") == "amd-x-2b"
    serve = next(s for s in plan_for(c) if s.name == "serve").cmds[0].argv
    assert "X=someone/else-sft:8005" in serve
    fwd = [a for a in P.tunnel_up(c).argv if a.startswith("127.0.0.1:")]
    assert fwd == [f"127.0.0.1:{p}:127.0.0.1:{p}" for p in range(8000, 8006)]
    assert argv_of(next(s for s in eval_steps(c) if s.name == "eval-L1").cmds[-1], "--run-tag") == "amd2-x"


def test_eval_only_names_are_validated_and_the_gate_subject_cannot_be_dropped():
    for bad in ((("A", "o/r"),), (("r", "o/r"),), (("R", "no-slash"),), (("R", "o/r"), ("R", "o/s"))):
        with pytest.raises(ValueError):
            cfg(eval_only=bad)
    with pytest.raises(ValueError, match="gate"):
        cfg(eval_only=(("X", "o/r"),))


def test_ports_are_fixed_by_model_so_dropping_an_arm_does_not_move_the_others():
    c = cfg(arms=("B",))
    assert [P.port_for(c, a) for a in ("base", "A", "B", "AB", "R")] == [8000, 8001, 8002, 8003, 8004]
    assert P.ports(c) == [8000, 8002, 8004]
    argv = P.tunnel_up(c).argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "-L"] == [
        f"127.0.0.1:{p}:127.0.0.1:{p}" for p in (8000, 8002, 8004)]


def test_the_tunnel_forwards_every_served_port_and_binds_loopback():
    argv = P.tunnel_up(cfg()).argv
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "-L"] == [
        f"127.0.0.1:{p}:127.0.0.1:{p}" for p in (8000, 8001, 8002, 8003, 8004)]
    assert "ExitOnForwardFailure=yes" in argv and "-N" in argv and "-f" in argv and "-S" in argv
    assert list(P.tunnel_down(cfg()).argv[-3:-1]) == ["-O", "exit"]


def test_the_gate_is_the_l1_sweep_on_a_fixed_subset_under_the_l1_run_tags():
    gate = P.gate_cmds(cfg())
    assert [argv_of(c, "--run-tag") for c in gate] == ["amd2-base", "amd2-r"]
    assert [argv_of(c, "--model") for c in gate] == ["amd-base-2b", "amd-r-2b"]
    for c in gate:
        assert argv_of(c, "--limit") == "60" and argv_of(c, "--rungs") == "L1"
        assert argv_of(c, "--samples") == "1" and argv_of(c, "--agent") == "bash"
        assert "--bash-stop" not in c.argv and "--no-climb" in c.argv
    l1 = {argv_of(c, "--run-tag"): c for c in next(s for s in eval_steps() if s.name == "eval-L1").cmds}

    def without_limit(cmd):
        i = cmd.argv.index("--limit")
        return cmd.argv[:i] + cmd.argv[i + 2:]
    for c in gate:    # same tag, same model, same flags but the limit: the 60 trials are reused
        full = l1[argv_of(c, "--run-tag")]
        assert without_limit(c) == without_limit(full) and c.env == full.env
    assert P.gate_limit(cfg(limit=40)) == 40


def test_the_gate_is_in_the_costed_plan_and_the_l1_row_does_not_pay_for_it_twice():
    rows = {r.stage: r for r in P.projection(cfg(), tokens(), P.Measured())}
    assert rows[P.ROW_GATE_SERVE].seconds > 0 and rows[P.ROW_GATE_EVAL].seconds > 0
    assert "done in the gate" in rows[P.ROW_L1].basis
    steps = {s.name: s for s in plan_for()}
    assert steps["gate-eval"].seconds == rows[P.ROW_GATE_EVAL].seconds
    assert steps["gate-serve"].billed and steps["gate-eval"].billed


def test_the_ssh_ready_probe_runs_a_real_command_and_the_wait_step_uses_it():
    cmd = P.ssh_ready(cfg())
    assert cmd.argv[-1] == "echo READY_$(whoami)"
    assert next(s for s in plan_for() if s.name == "wait-ssh").cmds[0] == cmd


def test_ssh_does_not_remember_host_keys_of_an_ephemeral_droplet_and_says_why():
    a = P.ssh(cfg(), ["true"]).argv
    assert "StrictHostKeyChecking=no" in a and "UserKnownHostsFile=/dev/null" in a
    assert "accept-new" not in " ".join(a)
    assert "ephemeral" in inspect_source(P.ssh_opts) and "reuse" in inspect_source(P.ssh_opts)


def test_ssh_is_batch_mode_and_keepalive_so_a_bad_key_fails_instead_of_hanging():
    a = P.ssh(cfg(), ["true"]).argv
    assert "BatchMode=yes" in a and "ServerAliveInterval=30" in a
    assert list(a[-2:]) == ["--", "true"]
    assert "-i" not in a and "-i" in P.ssh(cfg(identity="/k"), ["true"]).argv


# ═══ serve.sh: one merged server per model, run for real against stubs ═════════════

def stub_process_tools(tmp_path) -> tuple[Path, Path]:
    """pgrep that finds nothing and a pkill that only records its arguments: the scripts under test
    sweep for vLLM by command line, and a test must never touch a real process."""
    bin_dir, kills = tmp_path / "stubbin", tmp_path / "kills.txt"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (bin_dir / "pkill").write_text(f'#!/usr/bin/env bash\necho "$*" >> "{kills}"\nexit 1\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir, kills


@pytest.fixture
def serve_env(tmp_path):
    """A droplet-shaped tree with tiny REAL adapters for A, B, AB and a Hub adapter R, a tiny real
    base, the real merge script, and stubs for everything with a GPU or a network: the vLLM
    process (records its argv, answers /v1/models through a stub curl), the GPU's free memory, the
    Hub download and the probe."""
    import test_merge_adapter as TM
    root, log, stubs = tmp_path / "root", tmp_path / "log", tmp_path / "stubs"
    for d in (root / "runs", log, stubs):
        d.mkdir(parents=True)
    (root / "ops").symlink_to(REPO_ROOT / "ops")
    base_dir = tmp_path / "tinybase"
    base = TM.make_base(base_dir)
    for arm in ("a", "b", "ab"):
        TM.make_adapter(root / "runs" / f"sft_{arm}", base, TM.TARGETS)
        (root / "runs" / f"sft_{arm}" / ".done").write_text("done\n")     # a FINISHED arm
    TM.make_adapter(tmp_path / "hub_r", base, TM.TARGETS)
    (root / "data/train/sft_upstream").mkdir(parents=True)
    (root / "data/train/sft_upstream/train.jsonl").write_text("{}\n")
    started = tmp_path / "started.txt"
    # the "vLLM" the script starts: records argv, fails the first start of $STUB_FAIL_PORT, else
    # marks the port up and stays alive a few seconds
    syspy = stubs / "syspy"
    syspy.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        port=""; name=""
        while [[ $# -gt 0 ]]; do
          case "$1" in --port) port="$2";; --served-model-name) name="$2";; esac; shift
        done
        echo "$port $name" >> "{started}"
        if [[ "$port" == "${{STUB_FAIL_PORT:-none}}" ]]; then
          n=$(grep -c "^$port " "{started}")
          if (( n <= ${{STUB_FAIL_TIMES:-1}} )); then echo "Engine core initialization failed" >&2; exit 1; fi
        fi
        echo "$name" > "{stubs}/up.$port"
        exec sleep 6
        """))
    bin_dir, kills = stub_process_tools(tmp_path)
    (bin_dir / "curl").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        port=$(printf '%s' "$*" | sed -n 's|.*127.0.0.1:\\([0-9]*\\)/.*|\\1|p')
        [[ -f "{stubs}/up.$port" ]] || exit 22
        printf '{{"data": [{{"id": "%s"}}]}}' "$(cat "{stubs}/up.$port")"
        """))
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    probes = tmp_path / "probes.txt"
    (venv / "python").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        case "$*" in
          *mem_get_info*) echo "${{STUB_FREE:-0.97}}" ;;
          "- "*) cat > /dev/null; echo "$*" >> "{tmp_path}/downloads.txt"; echo "{tmp_path}/hub_r" ;;
          *ops.amd.probe_tools*)
            model=""; prev=""
            for a in "$@"; do [[ "$prev" == "--models" ]] && model="$a"; prev="$a"; done
            echo "$model" >> "{probes}"
            echo "TOOL_CALLS_OK=${{STUB_TOOLS:-1}}"
            echo "ADAPTER_CHECK model=$model differs=${{STUB_DIFFERS:-1}} tool_calls_ok=${{STUB_TOOLS:-1}}" ;;
          *) exec "{sys.executable}" "$@" ;;
        esac
        """))
    for f in (syspy, bin_dir / "curl", venv / "python"):
        f.chmod(0o755)
    (root / ".syspy").write_text(str(syspy) + "\n")
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "AMD_REMOTE_ROOT": str(root),
           "AMD_REMOTE_LOG": str(log), "AMD_VENV": str(tmp_path / "venv"), "AMD_HUB_NAMESPACE": "ns",
           "AMD_BASE_MODEL": str(base_dir), "HOME": str(tmp_path), "AMD_READY_SLEEP": "0.05",
           "AMD_GPU_FREE_SLEEP": "0", "AMD_GPU_ROOM_WAIT_S": "2"}
    return {"env": env, "root": root, "started": started, "tmp": tmp_path,
            "downloads": tmp_path / "downloads.txt", "probes": probes, "kills": kills}


HUB_R = "R=AdithyaSK/smoldataenvs-sft-2b-v0:8004"


def run_serve(se, *args, extra_env=None, check=True):
    out = subprocess.run(["bash", str(OPS / "serve.sh"), *args], env={**se["env"], **(extra_env or {})},
                         capture_output=True, text=True, timeout=120)
    if check:
        assert out.returncode == 0, out.stderr
    return out


def starts(se) -> list[tuple[str, str]]:
    if not se["started"].exists():
        return []
    return [tuple(line.split()) for line in se["started"].read_text().splitlines()]


def test_serve_sh_starts_one_server_per_model_on_fixed_ports_each_serving_a_merged_model(serve_env):
    run_serve(serve_env, "--wait", "--arms", "A,B,AB", "--hub", HUB_R)
    assert sorted(starts(serve_env)) == [("8000", "amd-base-2b"), ("8001", "amd-a-2b"), ("8002", "amd-b-2b"),
                                         ("8003", "amd-ab-2b"), ("8004", "amd-r-2b")]
    root = serve_env["root"]
    for name in ("amd-a-2b", "amd-b-2b", "amd-ab-2b", "amd-r-2b"):
        merged = root / "runs" / f"merged_{name}"
        assert json.loads((merged / "merge_report.json").read_text())["ok"]


def test_serve_sh_never_serves_lora_modules_which_do_not_work_for_this_model_on_vllm_0_17_1():
    code = [ln for ln in (OPS / "serve.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not any("--enable-lora" in ln or "--lora-modules" in ln for ln in code)


def record_argv(se):
    """Swap the stub 'vLLM' for one that dumps the argv the script builds."""
    stub = Path((se["root"] / ".syspy").read_text().strip())
    rec = se["tmp"] / "argv.txt"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{rec}"\nexit 0\n')
    return rec


def test_serve_sh_sets_the_tool_call_parser_the_non_thinking_template_loopback_and_the_gpu_share(serve_env):
    rec = record_argv(serve_env)
    run_serve(serve_env, "--arms", "A", "--hub", HUB_R)
    argv = rec.read_text().splitlines()
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert json.loads(argv[argv.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert {a for i, a in enumerate(argv) if argv[i - 1] == "--host"} == {"127.0.0.1"}
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == str(24 * 1024 ** 3)
    assert "--gpu-memory-utilization" not in argv and "--enable-prefix-caching" in argv
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert {a for i, a in enumerate(argv) if argv[i - 1] == "--tokenizer"} == {serve_env["env"]["AMD_BASE_MODEL"]}


def test_the_tool_call_parser_can_be_switched_without_editing_the_script(serve_env):
    rec = record_argv(serve_env)
    run_serve(serve_env, "--arms", "A", "--hub", HUB_R, extra_env={"AMD_TOOL_PARSER": "hermes"})
    argv = rec.read_text().splitlines()
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"


def test_serve_sh_with_fewer_arms_serves_fewer_models_and_keeps_the_other_ports(serve_env):
    run_serve(serve_env, "--wait", "--arms", "B", "--hub", HUB_R)
    assert sorted(starts(serve_env)) == [("8000", "amd-base-2b"), ("8002", "amd-b-2b"), ("8004", "amd-r-2b")]


def test_the_gate_serves_only_the_base_and_the_released_adapter_downloaded_from_the_hub(serve_env):
    run_serve(serve_env, "--wait", "--verify", "--arms", "", "--hub", HUB_R)
    assert sorted(starts(serve_env)) == [("8000", "amd-base-2b"), ("8004", "amd-r-2b")]
    assert "AdithyaSK/smoldataenvs-sft-2b-v0" in serve_env["downloads"].read_text()


def test_serve_prints_the_merge_report_verdict_and_the_adapter_check_for_every_adapter(serve_env):
    out = run_serve(serve_env, "--wait", "--verify", "--arms", "A", "--hub", HUB_R)
    parsed = driver.parse_serve(out.stdout + out.stderr)
    assert set(parsed["merge_checks"]) == {"amd-a-2b", "amd-r-2b"}
    assert parsed["merge_checks"]["amd-r-2b"] == {"modules_applied": 3, "tensors_changed": 3}
    assert parsed["adapter_checks"] == {"amd-a-2b": {"differs": True, "tool_calls_ok": True},
                                        "amd-r-2b": {"differs": True, "tool_calls_ok": True}}
    assert serve_env["probes"].read_text().split() == ["amd-a-2b", "amd-r-2b"]


def test_an_adapter_identical_to_the_base_fails_serve_so_it_cannot_be_evaluated(serve_env):
    out = run_serve(serve_env, "--wait", "--verify", "--arms", "A", "--hub", HUB_R,
                    extra_env={"STUB_DIFFERS": "0"}, check=False)
    assert out.returncode != 0 and "failed the check" in out.stderr
    assert "differs=0" in out.stdout


def test_an_adapter_with_no_tool_calls_fails_serve_too(serve_env):
    out = run_serve(serve_env, "--wait", "--verify", "--arms", "A", "--hub", HUB_R,
                    extra_env={"STUB_TOOLS": "0"}, check=False)
    assert out.returncode != 0 and "failed the check" in out.stderr


def test_the_base_servers_failed_start_waits_for_the_gpu_and_is_retried_once(serve_env):
    out = run_serve(serve_env, "--wait", "--arms", "", "--hub", HUB_R,
                    extra_env={"STUB_FAIL_PORT": "8000", "STUB_FAIL_TIMES": "1"})
    ports = [p for p, _ in starts(serve_env)]
    assert ports.count("8000") == 2 and ports.count("8004") == 1
    assert "exited during startup" in out.stderr and "restarting amd-base-2b" in out.stderr
    assert out.stderr.count("GPU room:") >= 2     # once before the first start, once before the retry


def test_a_server_that_fails_to_start_twice_stops_the_script_with_a_clear_message(serve_env):
    out = run_serve(serve_env, "--wait", "--arms", "", "--hub", HUB_R,
                    extra_env={"STUB_FAIL_PORT": "8000", "STUB_FAIL_TIMES": "9"}, check=False)
    assert out.returncode != 0 and "did not start after 2 attempts" in out.stderr
    assert [p for p, _ in starts(serve_env)].count("8000") == 2


def test_no_server_starts_until_the_gpu_has_given_its_memory_back(serve_env):
    out = run_serve(serve_env, "--wait", "--arms", "", "--hub", HUB_R, extra_env={"STUB_FREE": "0.2"},
                    check=False)
    assert out.returncode != 0 and "GPU memory not released" in out.stderr
    assert starts(serve_env) == []


def test_serve_merged_refuses_an_adapter_whose_merge_is_a_no_op_and_starts_no_server(serve_env):
    import test_merge_adapter as TM
    a = serve_env["root"] / "runs" / "sft_a"
    shutil.rmtree(a)
    TM.make_adapter(a, TM.make_base(serve_env["tmp"] / "other"), TM.TARGETS, zero_b=True)
    (a / ".done").write_text("done\n")
    out = run_serve(serve_env, "--wait", "--arms", "A", "--hub", HUB_R, check=False)
    assert out.returncode != 0 and "failed its checks" in out.stderr and "no-op" in out.stderr
    assert starts(serve_env) == []


def test_serve_merged_does_not_trust_a_stale_directory_that_holds_the_bases_weights(serve_env):
    root, base = serve_env["root"], Path(serve_env["env"]["AMD_BASE_MODEL"])
    merged = root / "runs" / "merged_amd-a-2b"
    shutil.copytree(base, merged)             # what the old merge left: config + the BASE's weights
    assert (merged / "config.json").exists()
    run_serve(serve_env, "--wait", "--arms", "A", "--hub", HUB_R)
    from ops.amd import merge_adapter as M
    assert M.check(merged) == [] and json.loads((merged / "merge_report.json").read_text())["tensors_changed"] == 3


def test_serve_stop_sweeps_stragglers_by_command_line_not_only_by_pidfile(serve_env):
    out = run_serve(serve_env, "--stop")
    assert "servers stopped" in out.stderr and "vllm" in serve_env["kills"].read_text()


def test_serve_names_and_ports_agree_between_the_scripts_and_the_plan():
    src = f"source {OPS}/common.sh; for m in base A B AB R X1; do amd_served_name $m; done; " \
          "for m in base A B AB; do amd_port $m; done"
    sh = subprocess.run(["bash", "-c", src], capture_output=True, text=True,
                        env={"PATH": os.environ["PATH"]}).stdout.split()
    c = cfg(eval_only=(("R", "o/r"), ("X1", "o/x")))
    assert sh[:6] == [P.served_name(m) for m in ("base", "A", "B", "AB", "R", "X1")]
    assert [int(x) for x in sh[6:]] == [P.port_for(c, m) for m in ("base", "A", "B", "AB")]


# ═══ run_sft.sh: the real script against a stub python ═══════════════════════════

@pytest.fixture
def fake_trainer(tmp_path):
    root, log, venv = tmp_path / "root", tmp_path / "log", tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (root / "data/train/sft_upstream").mkdir(parents=True)
    log.mkdir()
    (root / "data/train/sft_upstream/train.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (root / "data/train/sft_upstream/val.jsonl").write_text('{"a": 3}\n')
    (root / "data/train/ja3_sft_v2.jsonl").write_text('{"b": 1}\n')
    rec = tmp_path / "trainer-argv.txt"
    stub = venv / "bin" / "python"
    stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        case "$*" in
          *"ops.amd.resume finalize"*) echo x >> "{rec}.finalize" ;;
          *ops.amd.resume*) echo x >> "{rec}.status"
                            echo "{{\\"state\\": \\"${{STUB_STATE:-fresh}}\\", \\"step\\": null}}" ;;
          *mem_get_info*) echo "${{STUB_FREE:-0.97}}" ;;
          *ops.amd.sft_run*) printf '%s\\n' "$@" > "{rec}"; echo x >> "{rec}.runs"
                             runs=$(wc -l < "{rec}.runs")
                             if (( runs <= ${{STUB_FAIL_FIRST:-0}} )); then exit 1; fi ;;
          *) cat > /dev/null ;;
        esac
        """))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    bin_dir, _ = stub_process_tools(tmp_path)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "AMD_REMOTE_ROOT": str(root),
           "AMD_REMOTE_LOG": str(log), "AMD_VENV": str(venv), "AMD_HUB_NAMESPACE": "ns",
           "HOME": str(tmp_path), "AMD_GPU_FREE_SLEEP": "0", "AMD_GPU_FREE_TRIES": "2",
           "AMD_RESUME_SLEEP": "0", "AMD_GPU_ROOM_WAIT_S": "2"}
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
    assert argv[:2] == ["-m", "ops.amd.sft_run"]
    assert argv[argv.index("--data") + 1] == f"{root}/data/train/sft_upstream"
    assert argv[argv.index("--hub-model-id") + 1] == "ns/smol-ladder-sft-a-s2" and "--resume" in argv
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
    assert argv[argv.index("--data") + 1].endswith("data/train/ja3_sft_v2.jsonl")
    assert argv[argv.index("--hub-model-id") + 1] == "ns/smol-ladder-sft-b-s2"


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
        for block in re.findall(r"(?:train\.sft_lora|ops\.amd\.sft_run)(.*?)(?:\n\n|\)\n)", text, re.S):
            used |= set(re.findall(r'"?(--[a-z][a-z-]+)', block))
    used -= {"--save-steps"}          # the wrapper's own flag (ops/amd/sft_run.py), not the trainer's
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
    L.append(p, L.MEASURED, trials_per_min=18.5, tool_calls_ok=True, adapter_differs=True, gate_go=True)
    m = P.measured_from_ledger(L.read(p))
    assert (m.tokens_per_s, m.batch_size, m.grad_accum) == (9100.0, 4, 2)
    assert (m.trials_per_min, m.tool_calls_ok, m.gate_go, m.resume_ok) == (18.5, True, True, True)


# ═══ go / no-go ═════════════════════════════════════════════════════════════════════

DROPLET = 7


def events_for(tmp_path, **meas) -> tuple[P.Config, list[dict]]:
    c = cfg(ledger=str(tmp_path / "l.jsonl"), stage_dir=str(tmp_path / "stage"))
    (tmp_path / "stage").mkdir()
    (tmp_path / "stage" / "tokens.json").write_text(json.dumps({
        "max_length": 8192, "sets": {k: {"rows": v.rows, "trained_tokens": v.trained_tokens}
                                     for k, v in tokens().items()}}))
    fields = {"tokens_per_s": 9000.0, "batch_size": 8, "grad_accum": 1, "trials_per_min": 25.0,
              "tool_calls_ok": True, "adapter_differs": True, "gate_go": True, "checks_ok": True,
              "resume_ok": True}
    fields.update(meas)
    # a live droplet, and measurements stamped with ITS id and hardware (the only ones that count)
    L.append(c.ledger, L.CREATED, now=T0 - 60, price_per_hour=2.46, droplet_id=DROPLET)
    L.append(c.ledger, L.READY, now=T0 - 30, droplet_id=DROPLET, ip="198.51.100.7")
    L.append(c.ledger, L.MEASURED, now=T0, droplet_id=DROPLET, hardware=driver.hardware_of(c), **fields)
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
    assert not ok and any("tool calls" in r and "FAILED" in r for r in reasons)


def test_a_gate_that_did_not_pass_is_a_no_go_and_an_unmeasured_gate_is_too(tmp_path):
    for val, word in ((False, "FAILED"), (None, "not measured")):
        sub = tmp_path / str(val)
        sub.mkdir()
        c, ev = events_for(sub, gate_go=val)
        ok, reasons, _ = driver.go_no_go(c, ev, T0)
        assert not ok and any("the gate" in r and word in r for r in reasons)


def test_a_failed_resume_is_a_no_go(tmp_path):
    c, ev = events_for(tmp_path, resume_ok=False)
    assert not driver.go_no_go(c, ev, T0)[0]


def test_an_unmeasured_throughput_is_a_no_go_not_a_guess(tmp_path):
    c, ev = events_for(tmp_path, tokens_per_s=None)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("throughput" in r for r in reasons)


def test_a_projection_past_the_session_budget_stops_for_the_reviewer(tmp_path):
    c, ev = events_for(tmp_path, trials_per_min=1.0)       # the gate ran at 1 trial/min: evaluation alone is enormous
    ok, reasons, info = driver.go_no_go(c, ev, T0)
    assert not ok and any("exceeds the $35.00 budget" in r for r in reasons)
    assert info["remaining"] > 35.0


def test_money_already_spent_counts_toward_the_projection(tmp_path):
    c, ev = events_for(tmp_path)
    L.append(c.ledger, L.CREATED, now=T0, price_per_hour=2.46, droplet_id=DROPLET)
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

CLEAN = json.dumps({"agent_status": "exit 0", "reward": 1.0})


def make_runs(root: Path, tags, n=2):
    for tag in tags:
        for i in range(n):
            d = root / tag / "test" / f"task{i}" / "L1"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({"agent_status": "exit 0", "reward": 1.0}))


FINAL_FILES = ["adapter_config.json", "adapter_model.safetensors", "final.done"]


def local_final_adapters(logs: Path) -> str:
    """Final adapters for A, B and AB under logs/adapters (all the same bytes); returns their sha256."""
    for arm in ("A", "B", "AB"):
        (logs / "adapters" / arm).mkdir(parents=True, exist_ok=True)
        write_safetensors(logs / "adapters" / arm / "adapter_model.safetensors")
    return stage.sha256_file(logs / "adapters" / "A" / "adapter_model.safetensors")


def test_verify_sync_passes_when_adapters_checksums_and_runs_are_all_present(tmp_path, capsys):
    c = cfg(local_logs=str(tmp_path / "logs"), limit=2, samples=1)
    sha = local_final_adapters(tmp_path / "logs")
    f = tmp_path / "logs" / "adapters" / "A" / "adapter_model.safetensors"
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text(f"{stage.sha256_file(f)}  adapters/A/adapter_model.safetensors\n")
    make_runs(tmp_path / "runs", P.expected_trials(c))
    hub = lambda repo: FINAL_FILES  # noqa: E731
    assert driver.verify_sync(c, hub_files=hub, hub_sha256=lambda r, n: sha, runs_root=tmp_path / "runs",
                              namespace="ns") is True


def test_verify_sync_fails_if_an_adapter_is_not_readable_on_the_hub(tmp_path):
    c = cfg(local_logs=str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("")
    make_runs(tmp_path / "runs", P.expected_trials(c))
    assert driver.verify_sync(c, hub_files=lambda r: [], hub_sha256=lambda r, n: "", runs_root=tmp_path / "runs",
                              namespace="ns") is False


def test_verify_sync_fails_on_a_checksum_mismatch_and_on_a_missing_run_tag(tmp_path):
    c = cfg(local_logs=str(tmp_path / "logs"))
    (tmp_path / "logs" / "adapters").mkdir(parents=True)
    (tmp_path / "logs" / "adapters" / "x").write_bytes(b"changed")
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("0" * 64 + "  adapters/x\n")
    assert driver.sha_check(tmp_path / "logs") == ["adapters/x"]
    hub = lambda repo: FINAL_FILES  # noqa: E731
    sha = local_final_adapters(tmp_path / "logs")
    assert driver.verify_sync(c, hub_files=hub, hub_sha256=lambda r, n: sha, runs_root=tmp_path / "no-runs",
                              namespace="ns") is False


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
    (d / "ja3_sft_v2.jsonl").write_text((row + "\n") * 3)
    (d / "ja3_sft_v2.manifest.json").write_text(json.dumps({"replay": {"rate": 1.0}}))
    (d / "ja3_sft_v2.index.jsonl").write_text('{"task_id": "t"}\n' * 3)
    return tmp_path / "data"


def test_staging_arm_b_is_v2_and_refuses_v1_or_an_unproven_v2(tmp_path, monkeypatch):
    """Adapters trained on ja3_sft.jsonl (v1) are invalid: it is not the conversation the harness
    builds. Arm B is ja3_sft_v2, and staging refuses a v2 whose manifest does not record that every
    row replayed through the harness."""
    import tarfile

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    repo, data = make_repo(tmp_path), make_data(tmp_path)
    stage.stage(tmp_path / "ok", "HEAD", 8192, data, repo=repo, namespace="ns", encode=len)
    with tarfile.open(tmp_path / "ok" / "sft_b.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["ja3_sft_v2.index.jsonl", "ja3_sft_v2.jsonl",
                                          "ja3_sft_v2.manifest.json"]
    manifest = data / "train" / "ja3_sft_v2.manifest.json"
    manifest.write_text(json.dumps({"replay": {"rate": 0.99}}))
    with pytest.raises(SystemExit, match="replay"):
        stage.stage(tmp_path / "bad", "HEAD", 8192, data, repo=repo, namespace="ns", encode=len)
    manifest.write_text("{}")
    with pytest.raises(SystemExit, match="replay"):
        stage.stage(tmp_path / "bad2", "HEAD", 8192, data, repo=repo, namespace="ns", encode=len)
    (data / "train" / "ja3_sft_v2.jsonl").rename(data / "train" / "ja3_sft.jsonl")
    with pytest.raises(SystemExit, match="ja3_sft_v2.*v1.*invalid"):
        stage.stage(tmp_path / "v1", "HEAD", 8192, data, repo=repo, namespace="ns", encode=len)


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
    markers = ["stage.py", "driver.py preflight", "deadman.py", "driver.py create", "READY_$(whoami)",
               "scp -r", "entrypoint.sh", "smoke.sh", "serve.sh --wait --verify --arms ''",
               "--hub R=AdithyaSK/smoldataenvs-sft-2b-v0:8004", "tunnel", "--limit 60",
               "serve.sh --stop", "driver.py gate-decide", "driver.py project",
               "run_sft.sh --arm A", "run_sft.sh --arm B", "run_sft.sh --arm AB",
               "serve.sh --wait --verify --arms A,B,AB", "--rungs L1 --samples 1 --no-climb --limit 250", "--rungs L1 --samples 2",
               "--rungs L2", "--rungs L3", "--rungs L4", "amd2-base-program", "sync_back.sh",
               "verify-sync", "driver.py destroy"]
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
    text = (OPS / "container_setup.sh").read_text()
    assert '"torch", "torchvision", "torchaudio", "triton", "vllm"' in text
    assert "-c \"$LOGDIR/constraints.txt\"" in text and "--system-site-packages" in text


def test_the_server_is_never_bound_to_a_public_interface():
    assert "--host 127.0.0.1" in (OPS / "serve.sh").read_text()
    assert "0.0.0.0" not in " ".join(l for l in (OPS / "serve.sh").read_text().splitlines()
                                     if not l.lstrip().startswith("#"))


def test_the_hub_repos_are_created_private_before_training():
    text = (OPS / "container_setup.sh").read_text()
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


def test_the_runbook_quotes_the_totals_the_driver_computes():
    text = runbook()
    m = re.search(r"Costed plan: gate only \$([\d.]+)\. Gate \+ train \+ L1 eval \$([\d.]+)\. "
                  r"Everything bought \$([\d.]+)\.", text.replace("**", ""))
    assert m, "the runbook has no parseable costed-plan line"
    rows = P.projection(P.Config(), tokens_from_data(), P.Measured())
    st = P.staged_dollars(rows, P.PRICE_MI350X)
    assert [float(x) for x in m.groups()] == pytest.approx([st["gate_only"], st["core"], st["all"]], abs=0.006)
    m = re.search(r"\*\*TOTAL, every row\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\*", text)
    assert float(m.group(2)) == pytest.approx(P.total_dollars(rows, P.PRICE_MI350X), abs=0.006)


def tokens_from_data() -> dict[str, P.SetTokens]:
    """The same token counts the runbook quotes: from tokens.json if staged, else the numbers
    measured with the Qwen3.5 tokenizer and recorded in the runbook itself."""
    text = runbook()
    a = int(re.search(r"A ([\d,]+) trained tokens", text).group(1).replace(",", ""))
    b = int(re.search(r"B ([\d,]+) trained tokens", text).group(1).replace(",", ""))
    ab = int(re.search(r"AB ([\d,]+) \(A \+ B", text).group(1).replace(",", ""))
    assert ab == a + b
    return {"A": P.SetTokens(4439, a, "runbook"), "B": P.SetTokens(1122, b, "runbook"),
            "AB": P.SetTokens(5561, a + b, "runbook")}


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


# ═══ the gate's servers must be gone before an arm trains ═══════════════════════════

def test_the_plan_stops_the_gate_servers_after_the_gate_eval_and_before_the_go_no_go():
    steps = plan_for()
    n = names(steps)
    assert n.index("gate-eval") < n.index("stop-gate-server") < n.index("go-no-go") < n.index("sft-A")
    stop = steps[n.index("stop-gate-server")]
    assert list(stop.cmds[0].argv[-3:]) == ["bash", f"{cfg().remote_root}/ops/amd/serve.sh", "--stop"]
    assert stop.cmds[0].argv[-6:-3] == ("docker", "exec", "smol")
    assert stop.reserve is False      # never refused by the budget gate: it only ever saves money


def test_the_watchdog_counts_the_trainer_wrapper_as_work_so_a_long_arm_is_not_called_idle():
    pattern = re.search(r"WORK_PATTERN='([^']*)'", (OPS / "watchdog.sh").read_text()).group(1)
    assert re.search(pattern, "python -m ops.amd.sft_run --save-steps 50 --data x")
    assert re.search(pattern, "python -m vllm.entrypoints.openai.api_server --port 8000")


def test_run_sft_checkpoints_every_50_steps_by_default_and_the_interval_is_a_flag(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "A")
    assert argv[argv.index("--save-steps") + 1] == "50"
    argv = run_sft(fake_trainer, "--arm", "A", "--save-steps", "20")
    assert argv[argv.index("--save-steps") + 1] == "20"
    assert argv.index("--save-steps") < argv.index("--data")      # the wrapper's own flag comes first


def run_sft_raw(fake_trainer, *args, **env_over):
    env, rec, _, _ = fake_trainer
    out = subprocess.run(["bash", str(OPS / "run_sft.sh"), *args], env={**env, **env_over},
                         capture_output=True, text=True, timeout=60)
    runs = rec.with_name(rec.name + ".runs")
    return out, len(runs.read_text().splitlines()) if runs.exists() else 0


def test_a_crashed_run_is_resumed_automatically_after_the_card_recovers(fake_trainer):
    """A GPU reset killed one run in session 1. The loop re-checks the resume state (which sets a
    half-written checkpoint aside), waits for the card, and runs again with --resume."""
    out, runs = run_sft_raw(fake_trainer, "--arm", "A", STUB_FAIL_FIRST="1", AMD_MIN_PROGRESS_S="0")
    assert out.returncode == 0 and runs == 2
    assert "resuming (attempt 2 of 3)" in out.stderr
    status_calls = fake_trainer[1].with_name(fake_trainer[1].name + ".status").read_text().splitlines()
    assert len(status_calls) == 2                      # once to start, once before the second attempt
    assert "--resume" in fake_trainer[1].read_text().splitlines()


def test_the_resume_loop_is_bounded_by_max_attempts(fake_trainer):
    out, runs = run_sft_raw(fake_trainer, "--arm", "A", "--max-attempts", "2", STUB_FAIL_FIRST="9",
                            AMD_MIN_PROGRESS_S="0")
    assert out.returncode != 0 and runs == 2
    assert "2 attempt(s) are used up" in out.stderr


def test_a_run_that_dies_immediately_is_not_retried_because_it_is_not_a_gpu_reset(fake_trainer):
    out, runs = run_sft_raw(fake_trainer, "--arm", "A", STUB_FAIL_FIRST="9")     # default 120 s floor
    assert out.returncode != 0 and runs == 1 and "too early to be a GPU reset" in out.stderr


def test_the_resume_loop_waits_for_the_gpu_between_attempts(fake_trainer):
    # the first attempt crashes, then the card never comes back: the loop must not start a run into it
    out, runs = run_sft_raw(fake_trainer, "--arm", "A", STUB_FAIL_FIRST="1", AMD_MIN_PROGRESS_S="0",
                            STUB_FREE="0.15")
    assert out.returncode != 0 and runs == 0 and "GPU memory is not free" in out.stderr


def test_sft_run_forces_the_checkpoint_cadence_into_the_trainers_config_and_passes_the_rest_through(monkeypatch):
    import types
    seen = {}

    class Cfg:
        def __init__(self, **kw):
            seen["config"] = kw
    trl = types.ModuleType("trl")
    trl.SFTConfig = Cfg
    trainer = types.ModuleType("train.sft_lora")

    def main():
        seen["argv"] = list(sys.argv)
        import trl as t
        t.SFTConfig(save_steps=100, output_dir="x")          # what build() does
    trainer.main = main
    pkg = types.ModuleType("train")
    pkg.sft_lora = trainer
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "train", pkg)
    monkeypatch.setitem(sys.modules, "train.sft_lora", trainer)
    monkeypatch.setattr(sys, "argv", ["x"])
    from ops.amd import sft_run
    sft_run.main(["--save-steps", "25", "--data", "d", "--resume"])
    assert seen["config"] == {"save_steps": 25, "output_dir": "x"}
    assert seen["argv"] == ["train.sft_lora", "--data", "d", "--resume"]
    with pytest.raises(SystemExit):
        sft_run.main(["--save-steps", "0"])


def test_run_sft_stops_any_server_first_and_refuses_to_train_on_a_busy_gpu(fake_trainer, tmp_path):
    env, rec, _, _ = fake_trainer
    kills = tmp_path / "kills.txt"
    out = subprocess.run(["bash", str(OPS / "run_sft.sh"), "--arm", "A"], capture_output=True,
                         text=True, timeout=60, env={**env, "STUB_FREE": "0.15"})
    assert out.returncode != 0 and "GPU memory is not free" in out.stderr
    assert not rec.exists()                      # the trainer never started
    assert "vllm" in kills.read_text()           # but the sweep for a live server did run first


def test_run_sft_trains_when_the_gpu_is_free_and_says_so(fake_trainer):
    argv = run_sft(fake_trainer, "--arm", "A")
    assert argv[:2] == ["-m", "ops.amd.sft_run"]


def test_run_sft_on_a_finished_arm_does_not_need_a_free_gpu(fake_trainer):
    env, rec, _, _ = fake_trainer
    out = subprocess.run(["bash", str(OPS / "run_sft.sh"), "--arm", "A"], capture_output=True,
                         text=True, timeout=60, env={**env, "STUB_STATE": "done", "STUB_FREE": "0.1"})
    assert out.returncode == 0 and "already finished" in out.stderr


def test_the_watchdog_does_not_count_a_server_the_plan_has_stopped_as_work():
    # Documented in the plan: the idle trip counts vLLM as work, which is why it must be stopped.
    assert "vllm" in (OPS / "watchdog.sh").read_text()
    assert "stop-gate-server" in names(plan_for())


# ═══ sync_back.sh: runs for real, in a temp dir, with stubs ═════════════════════════

@pytest.fixture
def sync_env(tmp_path):
    root, log, venv = tmp_path / "root", tmp_path / "log", tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    root.mkdir()
    log.mkdir()
    (root / "tokens.json").write_text("{}")
    rec = tmp_path / "py.txt"
    stub = venv / "bin" / "python"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{rec}"\n')
    stub.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "AMD_REMOTE_ROOT": str(root), "AMD_REMOTE_LOG": str(log),
           "AMD_VENV": str(venv), "AMD_HUB_NAMESPACE": "ns", "HOME": str(tmp_path)}
    return env, root, log, rec


def run_sync(sync_env, *args):
    env, *_ = sync_env
    return subprocess.run(["bash", str(OPS / "sync_back.sh"), *args], env=env, capture_output=True,
                          text=True, timeout=60)


def test_sync_back_succeeds_when_no_arm_has_finished_and_adapters_does_not_exist(sync_env):
    _, root, log, _ = sync_env
    assert not (log / "adapters").exists()
    out = run_sync(sync_env, "--arms", "A,B,AB")
    assert out.returncode == 0, out.stderr
    assert "finished arms copied: none" in out.stderr and "sync complete" in out.stderr
    assert (log / "SHA256SUMS.artifacts").exists() and (log / "SHA256SUMS.artifacts").read_text() == ""
    assert (log / "tokens.json").exists()           # the logs still come home


def test_sync_back_still_pushes_the_logs_when_no_arm_has_finished(sync_env):
    _, _, _, rec = sync_env
    out = run_sync(sync_env, "--push-hub", "--arms", "A,B")
    assert out.returncode == 0, out.stderr
    calls = rec.read_text()
    assert "push_artifacts.py --repo ns/smol-ladder-runs-s2" in calls and "--verify" not in calls


def test_sync_back_copies_a_finished_arm_checksums_it_and_verifies_only_that_arm_on_the_hub(sync_env):
    _, root, log, rec = sync_env
    arm = root / "runs" / "sft_a"
    arm.mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors", ".done"):
        (arm / f).write_text("x")
    out = run_sync(sync_env, "--push-hub", "--arms", "A,B,AB")
    assert out.returncode == 0, out.stderr
    sums = (log / "SHA256SUMS.artifacts").read_text()
    assert "adapters/A/adapter_model.safetensors" in sums and "adapters/B" not in sums
    verify = [l for l in rec.read_text().splitlines() if "--verify" in l]
    assert verify == [f"{root}/ops/amd/push_artifacts.py --verify "
                      f"ns/smol-ladder-sft-a-s2={root}/runs/sft_a/adapter_model.safetensors"]


# ═══ bootstrap first contact ════════════════════════════════════════════════════════

def test_the_upload_is_idempotent_the_stage_dir_is_removed_first_and_waits_for_sshd():
    steps = plan_for()
    n = names(steps)
    assert n.index("create") < n.index("wait-ssh") < n.index("clean-stage") < n.index("upload") < n.index("bootstrap")
    clean = steps[n.index("clean-stage")].cmds[0].argv
    up = steps[n.index("upload")].cmds[0].argv
    assert list(clean[-4:]) == ["rm", "-rf", "--", cfg().remote_stage]
    assert up[-1].endswith(f":{cfg().remote_stage}")          # the same path the rm cleared


def test_a_remote_stage_that_is_not_a_safe_rm_target_is_refused_when_planning():
    for bad in ("/", "/tmp", "relative/path", "/var/../etc"):
        with pytest.raises(ValueError):
            plan_for(cfg(remote_stage=bad))


FIRST_BOOT = "Please wait while we get your droplet ready...\n"


def test_wait_for_ssh_runs_a_real_command_and_needs_its_exact_output():
    c, tries, naps = cfg(), [], []

    def runner(cmd, host, capture=False, timeout=None):
        tries.append((cmd, capture, timeout))
        # exit 0 with the first-boot banner is what `true` would have mistaken for ready
        return (0, FIRST_BOOT) if len(tries) < 4 else (0, FIRST_BOOT + "READY_root\n")
    assert driver.wait_for_ssh(c, runner=runner, sleep=naps.append, clock=lambda: 0.0) == 0
    assert len(tries) == 4 and naps == [10.0] * 3
    assert all(cmd == P.ssh_ready(c) and cap and t == 60.0 for cmd, cap, t in tries)


def test_ready_needs_the_exact_line_and_a_zero_exit():
    c = cfg()
    assert driver.ssh_is_ready(c, 0, "READY_root\n")
    assert not driver.ssh_is_ready(c, 0, FIRST_BOOT)
    assert not driver.ssh_is_ready(c, 0, "READY_ubuntu\n")             # another user: not our login
    assert not driver.ssh_is_ready(c, 255, "READY_root\n")
    assert not driver.ssh_is_ready(c, 0, "echo READY_root is what I will run")   # a mention is not the line


def test_wait_for_ssh_retries_after_a_timeout_kill_too():
    c, n = cfg(), []

    def runner(cmd, host, capture=False, timeout=None):
        n.append(1)
        return (driver.TIMEOUT_CODE, "") if len(n) < 3 else (0, "READY_root\n")
    assert driver.wait_for_ssh(c, runner=runner, sleep=lambda s: None, clock=lambda: 0.0) == 0 and len(n) == 3


def test_wait_for_ssh_gives_up_after_its_limit():
    clock = Clock(0.0)
    code = driver.wait_for_ssh(cfg(), runner=lambda *a, **k: (0, FIRST_BOOT), sleep=clock.sleep,
                               clock=clock, limit=100.0)
    assert code == 1 and clock.t >= 100.0


# ═══ entrypoint.sh ══════════════════════════════════════════════════════════════════

def test_apt_runs_under_a_lock_timeout_and_a_bounded_retry(tmp_path):
    bin_dir, count = tmp_path / "bin", tmp_path / "count"
    bin_dir.mkdir()
    (bin_dir / "apt-get").write_text(f'#!/usr/bin/env bash\necho "$*" >> "{count}"\n'
                                     f'[[ $(wc -l < "{count}") -ge 3 ]]\n')
    (bin_dir / "apt-get").chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "AMD_APT_SLEEP": "0"}
    ok = subprocess.run(["bash", "-c", f"source {OPS}/common.sh; amd_apt install -y curl"],
                        env=env, capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0                                # succeeded on the third attempt
    first = count.read_text().splitlines()[0]
    assert "DPkg::Lock::Timeout=180" in first and "install -y curl" in first
    count.write_text("x\n" * 0)
    (bin_dir / "apt-get").write_text(f'#!/usr/bin/env bash\necho "$*" >> "{count}"\nexit 100\n')
    bad = subprocess.run(["bash", "-c", f"source {OPS}/common.sh; amd_apt update"],
                         env={**env, "AMD_APT_TRIES": "4"}, capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0 and len(count.read_text().splitlines()) == 4     # bounded


def test_the_entrypoint_uses_the_bounded_apt_helper_everywhere():
    text = (OPS / "container_setup.sh").read_text()
    code = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    assert not any(re.search(r"(^|[;&|]\s*)apt-get ", l) for l in code)
    assert sum("amd_apt" in l for l in code) == 1     # the package install


def test_droplet_steps_run_in_the_smol_container_and_the_names_agree():
    steps = plan_for()
    smoke = steps[names(steps).index("smoke-checks")].cmds[0].argv
    i = smoke.index("docker")
    assert smoke[i:i + 4] == ("docker", "exec", P.CONTAINER, "bash")
    common = (OPS / "common.sh").read_text()
    assert f'AMD_CONTAINER="${{AMD_CONTAINER:-{P.CONTAINER}}}"' in common
    assert "--network host" in common and "--device /dev/kfd" in common and "--restart no" in common
    entry = (OPS / "entrypoint.sh").read_text()
    assert "amd_container_up" in entry and "amd_in_container" in entry and "watchdog.sh\" --arm" in entry
    assert "docker" not in (OPS / "smoke.sh").read_text()      # the steps themselves are container-agnostic
    assert "amd_in_container" in (OPS / "watchdog.sh").read_text()   # push in the container, poweroff on the host


def test_the_dataset_repo_is_asserted_private_even_when_it_pre_exists():
    assert "api.dataset_info(ds).private" in (OPS / "container_setup.sh").read_text()
    assert "dataset_info(repo).private" in (OPS / "push_artifacts.py").read_text()


def test_push_refuses_a_public_pre_existing_dataset_repo(tmp_path):
    from ops.amd import push_artifacts

    class Api:
        def create_repo(self, *a, **k): pass
        def dataset_info(self, repo): return type("I", (), {"private": False})()
    with pytest.raises(SystemExit, match="not private"):
        push_artifacts.push(Api(), "ns/runs", tmp_path)


def test_the_watchdog_is_started_detached_with_no_stdin():
    line = next(l for l in (OPS / "entrypoint.sh").read_text().splitlines() if "watchdog.sh\" --arm" in l)
    assert "setsid" in line and "</dev/null" in line


# ═══ the remote.env lives on the laptop only until the bootstrap succeeds ═══════════

def test_remote_env_is_deleted_from_the_laptop_stage_dir_after_a_successful_bootstrap(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "remote.env").write_text("HF_TOKEN=hf_secret\n")
    c = cfg(ledger=str(hb(tmp_path, time_now())), stage_dir=str(stage_dir),
            local_logs=str(tmp_path / "logs"))
    monkeypatch.setattr(driver, "run_cmd", lambda *a, **k: (0, ""))
    steps = [s for s in plan_for(c) if s.name == "bootstrap"]
    driver.run_steps(c, steps, None)
    assert not (stage_dir / "remote.env").exists()


def test_remote_env_survives_a_failed_bootstrap_so_it_can_be_retried(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "remote.env").write_text("HF_TOKEN=hf_secret\n")
    c = cfg(ledger=str(hb(tmp_path, time_now())), stage_dir=str(stage_dir),
            local_logs=str(tmp_path / "logs"))
    monkeypatch.setattr(driver, "run_cmd", lambda *a, **k: (1, ""))
    with pytest.raises(SystemExit):
        driver.run_steps(c, [s for s in plan_for(c) if s.name == "bootstrap"], None)
    assert (stage_dir / "remote.env").exists()


def test_uploading_without_remote_env_says_to_re_stage(tmp_path):
    c = cfg(stage_dir=str(tmp_path))
    with pytest.raises(SystemExit, match="stage.py"):
        driver.check_stage_for_upload(c)


def time_now() -> float:
    return __import__("time").time()


# ═══ resume: an arm that finished on an earlier droplet is skipped on a fresh one ═══

class FinalHub(HubStub):
    def __init__(self, files, tmp_path):
        super().__init__(None, tmp_path)
        self.files = files

    def list_files(self, repo):
        return list(self.files)


def test_an_arm_whose_final_adapter_is_on_the_hub_is_skipped_on_a_fresh_droplet(tmp_path):
    hub = FinalHub(["adapter_config.json", "adapter_model.safetensors", resume.HUB_DONE,
                    "last-checkpoint/trainer_state.json"], tmp_path)
    res = resume.status(tmp_path / "fresh-disk", "ns/sft-a", hub)
    assert res["state"] == "done" and res["source"] == "hub" and hub.downloads == 0


def test_adapter_files_on_the_hub_without_the_final_marker_are_just_a_checkpoint_and_training_resumes(tmp_path):
    # an adapter pushed by any strategy leaves a checkpoint's model files in the repo: not proof of "finished"
    hub = FinalHub(["adapter_config.json", "adapter_model.safetensors"], tmp_path)
    assert resume.status(tmp_path / "d", "ns/sft-a", hub)["state"] == "fresh"


def test_an_unreadable_hub_repo_never_skips_an_arm(tmp_path):
    class Broken(HubStub):
        def list_files(self, repo): raise OSError("hub down")
    assert resume.status(tmp_path / "d", "ns/sft-a", Broken(None, tmp_path))["state"] == "fresh"


def test_run_sft_hands_the_finish_to_the_verified_finalize_step():
    text = (OPS / "run_sft.sh").read_text()
    assert "ops.amd.resume finalize" in text and "(out / DONE).write_text" not in text


# ═══ verify-sync wants 90% of the trials, not one file ══════════════════════════════

def verified(tmp_path, n_files, limit=10, samples=1, ran=None, hub_ok=True):
    c = cfg(local_logs=str(tmp_path / "logs"), limit=limit, samples=samples)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("")
    runs = tmp_path / "runs"
    for tag in P.expected_trials(c):
        make_runs(runs, [tag], n=n_files)
    sha = local_final_adapters(tmp_path / "logs")
    hub = (lambda r: FINAL_FILES) if hub_ok else (lambda r: [])
    return driver.verify_sync(c, hub_files=hub, hub_sha256=lambda r, n: sha, runs_root=runs, namespace="ns",
                              ran=ran)


def test_a_run_tag_with_a_single_result_no_longer_passes(tmp_path):
    assert verified(tmp_path, 1) is False


def test_nine_of_ten_trials_passes_and_eight_of_ten_fails(tmp_path):
    assert verified(tmp_path, 9) is True
    (tmp_path / "x").mkdir()
    assert verified(tmp_path / "x", 8) is False


def test_only_the_first_rung_is_counted_because_later_rungs_skip_tasks_without_a_reference(tmp_path):
    c = cfg(limit=10, samples=1)
    assert driver.required_trials(c)["amd2-a"] == ("L1", 10, 9)
    root = tmp_path / "runs"
    for i in range(10):                       # L1 complete, L2 almost empty: still a pass
        (root / "amd2-a" / "test" / f"t{i}" / "L1").mkdir(parents=True)
        (root / "amd2-a" / "test" / f"t{i}" / "L1" / "result.json").write_text(CLEAN)
    (root / "amd2-a" / "test" / "t0" / "L2").mkdir(parents=True)
    (root / "amd2-a" / "test" / "t0" / "L2" / "result.json").write_text(CLEAN)
    assert driver.trial_files(root, "amd2-a", "test", "L1") == 10


def test_samples_land_in_s_k_directories_and_are_counted(tmp_path):
    root = tmp_path / "runs" / "t" / "test"
    for i in range(3):
        d = root / "task" / "L1" / f"s{i}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(CLEAN)
    (root / "task" / "L1" / "result.json").write_text(CLEAN)
    assert driver.trial_files(tmp_path / "runs", "t", "test", "L1") == 4


def test_a_harness_failure_is_not_a_trial_so_a_tag_of_timeouts_does_not_pass(tmp_path):
    root = tmp_path / "runs" / "t" / "test"
    for i in range(10):
        d = root / f"task{i}" / "L1"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"agent_status": "timeout" if i < 5 else "exit 0"}))
    assert driver.trial_files(tmp_path / "runs", "t", "test", "L1") == 5
    c = cfg(local_logs=str(tmp_path / "logs"), limit=10, samples=1)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text("")
    for tag in P.expected_trials(c):
        shutil.copytree(tmp_path / "runs" / "t", tmp_path / "runs" / tag)
    hub = lambda r: FINAL_FILES  # noqa: E731
    sha = local_final_adapters(tmp_path / "logs")
    assert driver.verify_sync(c, hub_files=hub, hub_sha256=lambda r, n: sha, runs_root=tmp_path / "runs",
                              namespace="ns") is False


def test_verify_after_an_l1_only_evaluation_does_not_demand_the_control_or_hint_rungs(tmp_path):
    c = cfg(limit=10, samples=1)
    assert set(driver.required_trials(c, ran={"eval-L1"})) == {"amd2-base", "amd2-a", "amd2-b", "amd2-ab",
                                                                "amd2-r"}
    assert set(driver.required_trials(c, ran={"eval-control"})) == {"amd2-base-program"}
    assert driver.required_trials(c, ran={"eval-L1", "eval-sample2"})["amd2-a"] == ("L1", 20, 18)
    assert driver.required_trials(c, ran=set()) == {}


# ═══ evaluation in stages: L1 first, then decide ════════════════════════════════════

def test_the_eval_stage_L1_keeps_only_the_five_L1_sweeps():
    steps = driver.select_eval(plan_for(), "L1")
    ev = [s for s in steps if s.phase == "eval"]
    assert [s.name for s in ev] == ["eval-L1"] and len(ev[0].cmds) == 5
    assert all("--rungs" in c.argv and c.argv[c.argv.index("--rungs") + 1] == "L1" for c in ev[0].cmds)
    assert "sft-A" in names(steps) and "destroy" in names(steps)        # nothing else is touched


def test_every_purchase_after_l1_is_a_separate_stage():
    def ev(stage):
        return [s.name for s in driver.select_eval(plan_for(), stage) if s.phase == "eval"]
    assert ev("sample2") == ["eval-sample2"]
    assert ev("hints") == ["eval-L2", "eval-L3", "eval-L4"]
    assert ev("control") == ["eval-control"]
    assert ev("rest") == ["eval-sample2", "eval-L2", "eval-L3", "eval-L4", "eval-control"]
    assert len(ev("all")) == 6


def test_the_default_stage_is_l1_only_for_eval_and_go():
    src = inspect_source(driver.main)
    assert 'default="L1"' in src and 'default="all"' not in src


def test_a_stage_that_selects_nothing_is_an_error():
    with pytest.raises(SystemExit):
        driver.select_eval(plan_for(cfg(rungs=("L1",))), "hints")


def test_eval_accepts_a_stage_flag_and_rejects_an_unknown_one(tmp_path):
    ok = run_driver(tmp_path, "eval", "--stage", "L1")
    assert "invalid choice" not in ok.stderr           # it parses; the gate then stops it (no deadman)
    assert "STOPPING BEFORE the evaluation" in ok.stderr and "no gate GO on record" in ok.stderr
    bad = run_driver(tmp_path, "eval", "--stage", "everything")
    assert bad.returncode == 2 and "invalid choice" in bad.stderr


def test_the_L1_step_costs_what_the_table_says_and_the_stages_add_up_to_the_whole():
    c = cfg()
    rows = {r.stage: r for r in P.projection(c, tokens(), P.Measured())}
    steps = {s.name: s for s in plan_for(c)}
    assert steps["eval-L1"].seconds == pytest.approx(rows[P.ROW_L1].seconds)
    assert steps["eval-sample2"].seconds == pytest.approx(rows[P.ROW_SAMPLE2].seconds)
    assert sum(steps[n].seconds for n in ("eval-L2", "eval-L3", "eval-L4")) == pytest.approx(
        rows[P.ROW_HINTS].seconds)
    assert steps["eval-control"].seconds == pytest.approx(rows[P.ROW_CONTROL].seconds)


def test_the_costed_table_separates_the_core_plan_from_the_optional_purchases():
    rows = P.projection(cfg(), tokens(), P.Measured())
    st = P.staged_dollars(rows, 2.46)
    by = {r.stage: r.dollars(2.46) for r in rows}
    assert st["hints"] == pytest.approx(by[P.ROW_HINTS]) and st["control"] == pytest.approx(by[P.ROW_CONTROL])
    assert st["core"] + st["sample2"] + st["hints"] + st["control"] == pytest.approx(st["all"])
    assert st["core"] < st["all"] and by[P.ROW_L1] > 0
    only_l1 = P.projection(cfg(rungs=("L1",), program_control=False, samples=2), tokens(), P.Measured())
    assert {P.ROW_HINTS, P.ROW_CONTROL, P.ROW_SAMPLE2}.isdisjoint({r.stage for r in only_l1})
    assert P.staged_dollars(only_l1, 2.46)["core"] == pytest.approx(P.total_dollars(only_l1, 2.46))


def test_the_printed_plan_shows_the_session_total_for_each_stopping_point_and_each_optional_stage(tmp_path):
    out = run_driver(tmp_path, "plan").stdout
    for needle in ("gate only (then destroy):", "gate + train + L1 eval + sync:", "+ L1 second sample:",
                   "+ L2-L4:", "+ program control:", "everything bought:"):
        assert needle in out, needle
    raw = run_driver(tmp_path, "plan", "--json").stdout
    table = json.loads(raw[raw.index("{"):])
    assert table["gate_only_dollars"] < table["core_dollars"] < table["total_dollars"]
    assert set(table["incremental_dollars"]) == {"sample2", "L2-L4", "control"}
    assert table["within_budget"] is True


# ═══ measurements and a GO are keyed to the droplet and the hardware ════════════════

def keyed_events(tmp_path, droplet_id=7, hw=None, **over):
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    L.append(c.ledger, L.CREATED, now=T0, price_per_hour=2.46, droplet_id=droplet_id)
    return c


def test_measurements_from_another_droplet_do_not_carry_over(tmp_path):
    c = keyed_events(tmp_path, droplet_id=7)
    L.append(c.ledger, L.MEASURED, now=T0, tokens_per_s=9000.0, droplet_id=6,
             hardware=driver.hardware_of(c))
    m = P.measured_from_ledger(L.read(c.ledger), driver.measure_key(c, L.read(c.ledger)))
    assert m.tokens_per_s == 0.0
    L.append(c.ledger, L.MEASURED, now=T0, tokens_per_s=8000.0, droplet_id=7,
             hardware=driver.hardware_of(c))
    m = P.measured_from_ledger(L.read(c.ledger), driver.measure_key(c, L.read(c.ledger)))
    assert m.tokens_per_s == 8000.0


def test_measurements_from_other_hardware_do_not_carry_over(tmp_path):
    c = keyed_events(tmp_path)
    fallback = cfg(ledger=c.ledger, size=P.SIZE_MI325X, region=P.REGION_MI325X, price=P.PRICE_MI325X)
    L.append(c.ledger, L.MEASURED, now=T0, tokens_per_s=9000.0, droplet_id=7,
             hardware=driver.hardware_of(fallback))
    assert P.measured_from_ledger(L.read(c.ledger), driver.measure_key(c, L.read(c.ledger))).tokens_per_s == 0.0


def test_a_stale_go_never_carries_over_to_a_new_droplet(tmp_path):
    c = keyed_events(tmp_path, droplet_id=7)
    L.append(c.ledger, L.MEASURED, now=T0, go=True, droplet_id=7, hardware=driver.hardware_of(c))
    ev = L.read(c.ledger)
    assert driver.last_go(ev, driver.measure_key(c, ev)) is True
    L.append(c.ledger, L.DESTROYED, now=T0 + HOUR)
    L.append(c.ledger, L.CREATED, now=T0 + 2 * HOUR, price_per_hour=2.46, droplet_id=8)
    ev = L.read(c.ledger)
    assert driver.last_go(ev, driver.measure_key(c, ev)) is None        # droplet 8 has no GO yet
    L.append(c.ledger, L.DESTROYED, now=T0 + 3 * HOUR)
    ev = L.read(c.ledger)
    assert driver.last_go(ev, driver.measure_key(c, ev)) is None        # nothing live: nothing is valid


def test_the_plan_after_a_destroy_shows_unmeasured_again(tmp_path):
    c = keyed_events(tmp_path)
    L.append(c.ledger, L.MEASURED, now=T0, tokens_per_s=9000.0, trials_per_min=5.0, droplet_id=7,
             hardware=driver.hardware_of(c))
    c.stage_dir = str(tmp_path / "none")
    _, rows_live = driver.make_plan(c, L.read(c.ledger))
    assert all(r.measured and r.basis.endswith("measured on this droplet")
               for r in rows_live if r.stage.startswith("sft"))
    L.append(c.ledger, L.DESTROYED, now=T0 + HOUR)
    _, rows_after = driver.make_plan(c, L.read(c.ledger))
    assert all(not r.measured and "session 1" in r.basis for r in rows_after if r.stage.startswith("sft"))


def test_every_measured_event_the_driver_writes_is_stamped_with_the_droplet_and_hardware():
    src = "".join(inspect_source(f) for f in (driver.run_steps, driver.cmd_project, driver.record_serve,
                                              driver.cmd_gate_decide, driver.run_supervised))
    assert src.count("**stamp(") >= 5
    assert driver.stamp(cfg(), [])["hardware"].startswith("gpu-mi350x1-288gb-spot|ric1|")


# ═══ status ═════════════════════════════════════════════════════════════════════════

def status_text(tmp_path, capsys, with_heartbeat, hours=1.0):
    now = __import__("time").time()
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    L.append(c.ledger, L.SESSION, now=now - hours * HOUR, budget=35.0, total_cap=90.0)
    L.append(c.ledger, L.CREATED, now=now - hours * HOUR, price_per_hour=2.46, droplet_id=77)
    L.append(c.ledger, L.READY, now=now - hours * HOUR + 60, droplet_id=77, ip="198.51.100.7")
    L.append(c.ledger, L.PRIOR, now=now - 2 * hours * HOUR, dollars=10.0)
    if with_heartbeat:
        hb(tmp_path, now)
    driver.print_status(c, L.summarise(L.read(c.ledger), now, 2.46), now)
    return capsys.readouterr().out


def test_status_prints_droplet_uptime_dollars_remaining_caps_and_the_hard_limit(tmp_path, capsys):
    out = status_text(tmp_path, capsys, with_heartbeat=True)
    assert "id 77" in out and "198.51.100.7" in out and "uptime 60.0 min" in out
    assert "session $2.46" in out and "total $12.46" in out
    assert "$32.54 under the $35.00 session budget" in out and "$77.54 under the $90.00 working total cap" in out
    assert "HARD limit $95.00" in out and "destroys at $89.50" in out
    assert "HEARTBEAT FRESH" in out


def test_status_says_loudly_when_there_is_no_live_deadman_and_how_to_start_one(tmp_path, capsys):
    out = status_text(tmp_path, capsys, with_heartbeat=False)
    assert "NO LIVE DEADMAN" in out and "setsid nohup python ops/amd/deadman.py" in out


# ═══ the runbook matches the fixes ══════════════════════════════════════════════════

def test_the_runbook_quotes_each_stopping_point_and_optional_stage_the_driver_computes():
    text = runbook().replace("**", "")
    rows = P.projection(P.Config(), tokens_from_data(), P.Measured())
    st = P.staged_dollars(rows, P.PRICE_MI350X)
    m = re.search(r"Gate only: \$([\d.]+)\. Gate \+ train \+ L1 eval: \$([\d.]+)\. The second L1 sample adds\s+"
                  r"\$([\d.]+)\. L2-L4 add \$([\d.]+)\.\s+The control adds \$([\d.]+)\.", text)
    assert m, "the runbook has no parseable per-stage line"
    assert [float(x) for x in m.groups()] == pytest.approx(
        [st["gate_only"], st["core"], st["sample2"], st["hints"], st["control"]], abs=0.006)


def test_the_runbook_table_rows_are_the_rows_the_driver_prints():
    text = runbook()
    for stage in (P.ROW_GATE_SERVE, P.ROW_GATE_EVAL, P.ROW_SERVE, P.ROW_L1, P.ROW_SAMPLE2, P.ROW_HINTS,
                  P.ROW_CONTROL):
        assert f"| {stage} |" in text, stage
    rows = {r.stage: r for r in P.projection(P.Config(), tokens_from_data(), P.Measured())}
    for stage, row in rows.items():
        m = re.search(r"\| %s \| ([\d.]+) \| ([\d.]+) \|" % re.escape(stage), text)
        assert m, stage
        assert float(m.group(2)) == pytest.approx(row.dollars(P.PRICE_MI350X), abs=0.006), stage
        assert float(m.group(1)) == pytest.approx(row.seconds / 3600.0, abs=0.006), stage
    assert "serve + evaluate (4 models)" not in text


def test_the_runbook_states_the_hard_limit_the_detached_start_and_the_rearm_rule():
    text = runbook()
    assert "$95 HARD limit" in text and "HEARTBEAT FRESH" in text
    assert "setsid nohup python ops/amd/deadman.py" in text and "< /dev/null &" in text
    assert "**Re-arm it before any further create.**" in text or "RE-ARM before any further create" in text
    assert "--stage L1" in text and "--stage rest" in text and "stop-gate-server" in text
    assert "final.done" in text and "at least 90%" in text


def test_the_runbook_command_block_runs_the_deadman_before_create_and_L1_before_the_hint_rungs():
    text = runbook()
    block = text[text.index("```sh", text.index("## 4.")):]
    block = block[:block.index("```", 6)]
    order = ["deadman.py", "driver.py status", "driver.py create", "driver.py smoke",
             "driver.py train", "driver.py serve", "eval --stage L1", "eval --stage rest",
             "driver.py sync", "driver.py destroy"]
    pos = [block.index(c) for c in order]
    assert pos == sorted(pos), dict(zip(order, pos))


def test_every_command_the_runbook_block_shows_is_one_the_driver_parses(tmp_path):
    text = runbook()
    block = text[text.index("```sh", text.index("## 4.")):]
    block = block[:block.index("```", 6)]
    for m in re.finditer(r"driver\.py (eval --stage \w+|status|plan|dry-run)", block):
        argv = m.group(1).split()
        out = run_driver(tmp_path, *argv)
        assert "invalid choice" not in out.stderr and "unrecognized" not in out.stderr, m.group(0)


# ═══ the probe must see the adapter change the model's output ══════════════════════════

def test_an_adapter_whose_output_equals_the_bases_is_a_no_go(tmp_path):
    c, ev = events_for(tmp_path, adapter_differs=False)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("differs from the base" in r and "FAILED" in r for r in reasons)


def test_an_unmeasured_adapter_check_is_a_no_go(tmp_path):
    c, ev = events_for(tmp_path, adapter_differs=None)
    ok, reasons, _ = driver.go_no_go(c, ev, T0)
    assert not ok and any("differs from the base" in r and "not measured" in r for r in reasons)


def test_parse_serve_reads_the_merge_verdicts_and_the_adapter_checks():
    out = ("MERGE_OK model=amd-r-2b modules_applied=204 tensors_changed=204 max_relative_delta=0.0123\n"
           "TOOL_CALLS_OK=1\nADAPTER_DIFFERS_FROM_BASE=1\n"
           "ADAPTER_CHECK model=amd-r-2b differs=1 tool_calls_ok=1\n"
           "ADAPTER_CHECK model=amd-a-2b differs=0 tool_calls_ok=1\n")
    assert driver.parse_serve(out) == {
        "adapter_checks": {"amd-r-2b": {"differs": True, "tool_calls_ok": True},
                           "amd-a-2b": {"differs": False, "tool_calls_ok": True}},
        "merge_checks": {"amd-r-2b": {"modules_applied": 204, "tensors_changed": 204}}}


def _rows_file(tmp_path):
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                        {"role": "assistant", "content": "", "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "ls -la /home/user/input"}}}]},
                        {"role": "tool", "content": "x"}],
           "tools": [{"type": "function", "function": {"name": "bash"}}]}
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(row) + "\n")
    return p


def test_the_fixed_prompt_is_the_first_training_row_up_to_its_first_assistant_turn(tmp_path):
    prompt, tools, target = probe_tools.first_training_prompt(_rows_file(tmp_path))
    assert [m["role"] for m in prompt] == ["system", "user"] and tools[0]["function"]["name"] == "bash"
    assert target == "ls -la /home/user/input"


def test_token_agreement_is_positional_over_the_targets_tokens():
    assert probe_tools.token_agreement("ls -la /home/user/input", "ls -la /home/user/input") == 1.0
    assert probe_tools.token_agreement("ls /home/user/input", "ls -la /home/user/input") < 0.5
    assert probe_tools.token_agreement("", "ls -la") == 0.0


def test_adapter_check_flags_identical_output_and_reports_agreement(tmp_path, monkeypatch):
    def reply(text):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "bash", "arguments": json.dumps({"command": text})}}]},
            "logprobs": {"content": [{"logprob": -0.5}]}}]}
    outs = {"adapter": "ls -la /home/user/input", "base": "head -5 data.csv"}
    monkeypatch.setattr(probe_tools, "chat", lambda port, body, timeout=120.0: reply(outs[body["model"]]))
    rows = _rows_file(tmp_path)
    res = probe_tools.adapter_check(1, "adapter", 1, "base", rows)
    assert res["differs"] and res["adapter_agreement"] == 1.0 and res["base_agreement"] < 1.0
    outs["adapter"] = outs["base"]
    assert probe_tools.adapter_check(1, "adapter", 1, "base", rows)["differs"] is False


# ═══ the gate: the released adapter through the whole stack, before anything is trained ═════════

from ops.amd import gate as G  # noqa: E402
from ops.amd import supervise as SUP  # noqa: E402


def result(reward=1.0, status="exit 0", stop="model_stopped", pred="1"):
    return {"agent_status": status, "reward": reward, "stop_reason": stop, "prediction": pred}


def tag_dir(root: Path, tag: str, results: dict, split="test"):
    """results: task id -> result dict, written the way the harness lays them out (sample 0, L1)."""
    for task, res in results.items():
        d = root / tag / split / task / "L1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res))


def stats(arm, results, n=None):
    ids = [f"t{i}" for i in range(n if n is not None else len(results))]
    return G.stats_for(arm, {f"t{i}": r for i, r in enumerate(results)}, ids)


def decide(base, adapter, **over):
    kw = dict(adapter_name="R", differs=True, tool_calls_ok=True, merge_ok=True, margin=0.05,
              max_failures=3)
    kw.update(over)
    return G.decide(base, adapter, **kw)


def mixed(n_pass, n, **kw):
    return [result(1.0 if i < n_pass else 0.0, **kw) for i in range(n)]


def test_the_gate_is_go_when_the_stack_is_healthy_and_the_released_adapter_beats_the_base():
    d = decide(stats("base", mixed(10, 60)), stats("R", mixed(30, 60)))
    assert d.go and not d.needs_accept and not d.hard_failures
    assert "### GATE: GO" in d.lines[-1]


def test_harness_failures_above_the_tolerance_are_a_hard_no_go_that_accept_gate_cannot_override():
    bad = mixed(30, 55) + [result(status="timeout")] * 5
    d = decide(stats("base", mixed(10, 60)), stats("R", bad), accepted=True)
    assert not d.go and not d.needs_accept and "R: 5 harness failures of 60" in d.hard_failures[0]
    ok = mixed(30, 57) + [result(status="timeout")] * 3          # exactly the tolerance
    assert decide(stats("base", mixed(10, 60)), stats("R", ok)).go


def test_a_trial_that_was_never_written_counts_as_a_failure_not_as_a_smaller_sample():
    # R produced 40 of 60 results: a sweep that died, which looks like "40 trials, all fine" if only
    # the files that exist are counted
    ids = [f"t{i}" for i in range(60)]
    r_results = {f"t{i}": result() for i in range(40)}
    d = decide(stats("base", mixed(10, 60)), G.stats_for("R", r_results, ids))
    assert not d.go and any("20 harness failures" in h for h in d.hard_failures)


def test_an_adapter_that_is_identical_to_the_base_is_a_no_go_even_with_accept_gate():
    d = decide(stats("base", mixed(10, 60)), stats("R", mixed(30, 60)), differs=False, accepted=True)
    assert not d.go and not d.needs_accept
    assert any("differs from the base's: FAILED" in h for h in d.hard_failures)
    assert any("not measured" in h for h in decide(stats("base", mixed(1, 60)), stats("R", mixed(30, 60)),
                                                   differs=None).hard_failures)


def test_a_merge_report_that_did_not_pass_or_no_tool_call_is_a_no_go():
    base, ad = stats("base", mixed(10, 60)), stats("R", mixed(30, 60))
    assert any("merge report" in h for h in decide(base, ad, merge_ok=False).hard_failures)
    assert any("tool call" in h for h in decide(base, ad, tool_calls_ok=False).hard_failures)


def test_a_rate_below_base_plus_margin_prints_the_comparison_and_needs_accept_gate():
    base, ad = stats("base", mixed(20, 60)), stats("R", mixed(22, 60))     # +3.3 points, margin 5
    d = decide(base, ad)
    assert not d.go and d.needs_accept and not d.hard_failures
    text = "\n".join(d.lines)
    assert "pass rate: base 0.333, R 0.367" in text and "required +0.050" in text
    assert "paired on 60 tasks" in text and "exact sign test" in text
    assert "--accept-gate" in d.lines[-1]
    ok = decide(base, ad, accepted=True)
    assert ok.go and ok.accepted and "ACCEPTED" in ok.lines[-1]


def test_the_paired_comparison_counts_each_cell_and_the_exact_sign_test():
    base = stats("base", [result(1), result(1), result(0), result(0), result(1)])
    other = stats("R", [result(1), result(0), result(1), result(1), result(1)])
    p = G.paired(base, other)
    assert (p["both"], p["only_base"], p["only_other"], p["neither"]) == (2, 1, 2, 0)
    assert p["p_value"] == pytest.approx(1.0)                  # 1 vs 2 discordant pairs: no evidence
    wide = G.paired(stats("b", [result(0)] * 12), stats("o", [result(1)] * 12))
    assert wide["only_other"] == 12 and wide["p_value"] == pytest.approx(2 / 2 ** 12)
    assert G.paired(stats("b", []), stats("o", []))["p_value"] == 1.0


def test_harness_failures_are_left_out_of_the_paired_rates_not_counted_as_model_failures():
    base = stats("base", [result(1), result(status="timeout"), result(0)])
    other = stats("R", [result(1), result(1), result(1)])
    p = G.paired(base, other)
    assert p["tasks"] == 2 and p["base_rate"] == 0.5 and p["other_rate"] == 1.0


def test_the_stop_reason_histograms_show_each_models_share_of_the_three_endings_that_matter():
    ad = stats("R", [result(stop="answer_submitted")] * 3 + [result(stop="max_turns", pred="")] * 5
               + [result(stop="context_exhausted", pred="")] * 2)
    base = stats("base", [result(stop="model_stopped")] * 4 + [result(stop="max_turns", pred="")] * 6)
    d = decide(stats("base", mixed(1, 10)), ad, max_failures=0)
    text = "\n".join(d.lines)
    assert "answer_submitted 3 (30%)" in text and "max_turns 5 (50%)" in text
    assert "context_exhausted 2 (20%)" in text and "ended with an answer 3 (30%)" in text
    assert "WARNING: R runs out of turns in at least half its trials" in text
    assert ad.stop_histogram() == {"answer_submitted": 3, "max_turns": 5, "context_exhausted": 2}
    assert base.ended_with_answer() == 4


def test_the_gate_reads_sample_zero_of_l1_for_a_task_subset_only(tmp_path):
    root = tmp_path / "runs"
    tag_dir(root, "x", {"t0": result(), "t1": result(0.0), "t2": result()})
    d = root / "x" / "test" / "t0" / "L1" / "s1"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(result(0.0)))         # sample 1 is not the gate's
    (root / "x" / "test" / "t0" / "L2").mkdir()
    (root / "x" / "test" / "t0" / "L2" / "result.json").write_text("{}")
    got = G.read_results(root, "x", "test", task_ids={"t0", "t1"})
    assert set(got) == {"t0", "t1"} and G.passed(got["t0"]) and not G.passed(got["t1"])
    assert G.task_ids_present(root, ["x", "y"], "test") == ["t0", "t1", "t2"]
    (root / "x" / "test" / "t3" / "L1").mkdir(parents=True)
    (root / "x" / "test" / "t3" / "L1" / "result.json").write_text("{not json")
    assert G.is_clean(G.read_results(root, "x", "test")["t3"]) is False


def gate_ledger(tmp_path, **extra):
    """A live droplet with the gate's serve checks recorded (stamped), as run_steps writes them."""
    c = cfg(ledger=str(tmp_path / "l.jsonl"), local_logs=str(tmp_path / "logs"))
    L.append(c.ledger, L.CREATED, now=T0 - 60, price_per_hour=2.46, droplet_id=DROPLET)
    L.append(c.ledger, L.READY, now=T0 - 30, droplet_id=DROPLET, ip="198.51.100.7")
    key = {"droplet_id": DROPLET, "hardware": driver.hardware_of(c)}
    fields = {"serve_gate": True, "merge_checks": {"amd-r-2b": {"modules_applied": 3, "tensors_changed": 3}},
              "adapter_checks": {"amd-r-2b": {"differs": True, "tool_calls_ok": True}},
              "tool_calls_ok": True, "adapter_differs": True}
    fields.update(extra)
    L.append(c.ledger, L.MEASURED, now=T0, **fields, **key)
    return c


def test_gate_decide_reads_the_results_and_the_serve_checks_and_records_a_stamped_verdict(tmp_path, capsys):
    c = gate_ledger(tmp_path)
    root = tmp_path / "runs"
    ids = [f"t{i}" for i in range(60)]
    tag_dir(root, P.run_tag(c, "base"), {t: result(1.0 if i < 10 else 0.0) for i, t in enumerate(ids)})
    tag_dir(root, P.run_tag(c, "R"), {t: result(1.0 if i < 30 else 0.0) for i, t in enumerate(ids)})
    assert driver.cmd_gate_decide(c, runs_root=root) is True
    out = capsys.readouterr().out
    assert "### GATE: GO" in out and "base vs R on 60 tasks" in out
    last = L.read(Path(c.ledger))[-1]
    assert last["gate_go"] is True and last["droplet_id"] == DROPLET and "hardware" in last
    assert P.measured_from_ledger(L.read(Path(c.ledger)), driver.measure_key(c, L.read(Path(c.ledger)))).gate_go


def test_gate_decide_judges_the_recorded_gate_subset_even_after_the_full_l1_run_wrote_more(tmp_path):
    c = gate_ledger(tmp_path)
    root = tmp_path / "runs"
    gate_ids = [f"t{i}" for i in range(60)]
    all_ids = [f"t{i}" for i in range(250)]
    tag_dir(root, P.run_tag(c, "base"), {t: result(1.0 if i < 10 else 0.0) for i, t in enumerate(all_ids)})
    tag_dir(root, P.run_tag(c, "R"), {t: result(1.0 if i < 30 else 0.0) for i, t in enumerate(all_ids)})
    Path(c.local_logs).mkdir()
    (Path(c.local_logs) / "gate_tasks.json").write_text(json.dumps(gate_ids))
    assert driver.gate_task_ids(c, root) == gate_ids
    assert driver.cmd_gate_decide(c, runs_root=root) is True


def test_gate_decide_with_no_results_or_a_failed_adapter_check_is_a_no_go_and_is_recorded(tmp_path, capsys):
    c = gate_ledger(tmp_path, adapter_differs=False)
    root = tmp_path / "runs"
    tag_dir(root, P.run_tag(c, "base"), {"t0": result(0.0)})
    tag_dir(root, P.run_tag(c, "R"), {"t0": result(1.0)})
    assert driver.cmd_gate_decide(c, runs_root=root) is False
    assert "NO-GO" in capsys.readouterr().out
    assert L.read(Path(c.ledger))[-1]["gate_go"] is False


def test_the_gate_stops_on_a_rate_that_did_not_clear_the_margin_until_accept_gate_is_given(tmp_path, capsys):
    c = gate_ledger(tmp_path)
    root = tmp_path / "runs"
    ids = [f"t{i}" for i in range(60)]
    for arm in ("base", "R"):
        tag_dir(root, P.run_tag(c, arm), {t: result(1.0 if i < 20 else 0.0) for i, t in enumerate(ids)})
    assert driver.cmd_gate_decide(c, runs_root=root) is False
    assert "STOP" in capsys.readouterr().out
    c.accept_gate = True
    assert driver.cmd_gate_decide(c, runs_root=root) is True
    assert "ACCEPTED" in capsys.readouterr().out


def test_gate_decide_is_a_cli_command_that_exits_nonzero_without_a_gate(tmp_path):
    out = run_driver(tmp_path, "gate-decide")
    assert out.returncode == 3 and "NO-GO" in out.stdout


def test_accept_gate_and_the_gate_flags_parse(tmp_path):
    out = run_driver(tmp_path, "gate-decide", "--accept-gate", "--gate-margin", "0.1",
                     "--gate-max-failures", "1", "--gate-tasks", "30")
    assert "unrecognized" not in out.stderr and out.returncode == 3


def test_the_gate_decision_is_a_step_of_smoke_and_a_command_of_its_own():
    src = inspect_source(driver.main)
    assert '"smoke": ("smoke", "gate")' in src and '"gate": ("gate",)' in src and 'c == "gate-decide"' in src


# ═══ the evaluation cannot start without the guards having passed ═════════════════════════════

def eval_ledger(tmp_path, gate_go=True, **serve_over):
    c = gate_ledger(tmp_path)
    key = {"droplet_id": DROPLET, "hardware": driver.hardware_of(c)}
    if gate_go is not None:
        L.append(c.ledger, L.MEASURED, now=T0 + 1, gate_go=gate_go, **key)
    serve = {"merge_checks": {P.served_name(a): {"modules_applied": 3, "tensors_changed": 3}
                              for a in P.served_adapters(c)},
             "adapter_checks": {P.served_name(a): {"differs": True, "tool_calls_ok": True}
                                for a in P.served_adapters(c)}}
    serve.update(serve_over)
    if serve is not None:
        L.append(c.ledger, L.MEASURED, now=T0 + 2, **serve, **key)
    return c


def blockers(c):
    return driver.eval_blockers(c, L.read(Path(c.ledger)))


def test_the_evaluation_may_start_after_a_gate_go_and_a_serve_that_checked_every_adapter(tmp_path):
    assert blockers(eval_ledger(tmp_path)) == []


def test_no_gate_go_blocks_the_evaluation_even_with_a_verified_serve(tmp_path):
    for val in (None, False):
        sub_ = tmp_path / str(val)
        sub_.mkdir()
        why = blockers(eval_ledger(sub_, gate_go=val))
        assert len(why) == 1 and "no gate GO on record" in why[0]


def test_a_gate_go_from_another_droplet_does_not_count(tmp_path):
    c = eval_ledger(tmp_path)
    L.append(c.ledger, L.DESTROYED, now=T0 + 5)
    L.append(c.ledger, L.CREATED, now=T0 + 10, price_per_hour=2.46, droplet_id=8)
    why = blockers(c)
    assert any("no gate GO" in w for w in why) and any("no `serve` on record" in w for w in why)


def test_every_served_adapter_needs_a_merge_report_a_differing_output_and_a_tool_call(tmp_path):
    c = eval_ledger(tmp_path)
    key = {"droplet_id": DROPLET, "hardware": driver.hardware_of(c)}
    L.append(c.ledger, L.MEASURED, now=T0 + 3, **key,
             merge_checks={"amd-a-2b": {}, "amd-b-2b": {}, "amd-ab-2b": {}},    # R has no report
             adapter_checks={"amd-a-2b": {"differs": False, "tool_calls_ok": True},
                             "amd-b-2b": {"differs": True, "tool_calls_ok": False},
                             "amd-ab-2b": {"differs": True, "tool_calls_ok": True},
                             "amd-r-2b": {"differs": True, "tool_calls_ok": True}})
    why = "\n".join(blockers(c))
    assert "amd-r-2b: no passing merge report" in why
    assert "amd-a-2b: its temperature-0 output is IDENTICAL to the base's" in why
    assert "amd-b-2b: no parsed bash tool call" in why and "amd-ab-2b" not in why


def test_the_gates_own_serve_record_does_not_satisfy_the_full_serve_requirement(tmp_path):
    c = gate_ledger(tmp_path)
    key = {"droplet_id": DROPLET, "hardware": driver.hardware_of(c)}
    L.append(c.ledger, L.MEASURED, now=T0 + 1, gate_go=True, **key)
    why = "\n".join(blockers(c))
    assert "no `serve` on record" in why


def test_run_steps_refuses_the_eval_phase_without_the_guards_and_starts_nothing(tmp_path, monkeypatch):
    c = gate_ledger(tmp_path)
    called = []
    monkeypatch.setattr(driver, "run_supervised", lambda *a, **k: called.append(1) or 0)
    steps = [s for s in plan_for(c) if s.name == "eval-L1"]
    with pytest.raises(SystemExit, match="STOPPING BEFORE the evaluation"):
        driver.run_steps(c, steps, only="eval")
    assert called == []


def test_record_serve_stamps_the_released_adapters_verdicts_only_for_the_gate(tmp_path):
    c = gate_ledger(tmp_path)
    out = ("MERGE_OK model=amd-r-2b modules_applied=204 tensors_changed=204 max_relative_delta=0.01\n"
           "ADAPTER_CHECK model=amd-r-2b differs=0 tool_calls_ok=1\n")
    driver.record_serve(c, out, gate=True)
    ev = L.read(Path(c.ledger))[-1]
    assert ev["serve_gate"] is True and ev["adapter_differs"] is False and ev["tool_calls_ok"] is True
    assert ev["droplet_id"] == DROPLET
    driver.record_serve(c, out, gate=False)
    ev = L.read(Path(c.ledger))[-1]
    assert ev["serve_gate"] is False and "adapter_differs" not in ev


def test_the_smoke_and_the_gate_steps_pass_serve_output_to_record_serve():
    src = inspect_source(driver.run_steps)
    assert 'step.name in ("gate-serve", "serve")' in src and "record_serve(" in src
    assert driver.CAPTURED == ("gate-serve", "serve", "smoke-checks")


# ═══ supervision: a stalled evaluation costs minutes, not hours ═══════════════════════════════

class FakeHandle:
    def __init__(self, sim, i):
        self.sim, self.i, self.code = sim, i, None

    def poll(self):
        return self.code

    def stop(self, grace=0):
        if self.code is None:
            self.code = -15
            self.sim.stopped.append(self.i)


class Sim:
    """Drives a Supervisor with no process, socket or sleep: a fake wall clock whose sleeps run a
    per-tick script that writes result files (with mtimes from that clock) and ends handles."""

    def __init__(self, tmp_path, n=2, expected=20, script=(), **over):
        self.root = tmp_path / "runs"
        self.clock = Clock(1_000_000.0)
        self.script = list(script)
        self.stopped, self.launches, self.restart_calls, self.logs, self.probes = [], [], [], [], []
        self.health = {}
        self.restart_ok = True
        self.n_written = {}
        self.targets = [SUP.Target(f"m{i}", f"tag{i}", "test", "L1", 8000 + i, expected) for i in range(n)]
        kw = dict(name="eval-L1", targets=self.targets, results_root=self.root, launch=self.launch,
                  restart_servers=self.restart, log=self.logs.append, accrued=lambda: 1.5,
                  probe=self.probe, clock=self.clock, sleep=self.sleep, tick_s=60.0, stall_s=300.0)
        kw.update(over)
        self.sup = SUP.Supervisor(**kw)

    def launch(self, retry):
        self.launches.append(retry)
        return [FakeHandle(self, i) for i in range(len(self.targets))]

    def restart(self):
        self.restart_calls.append(self.clock.t)
        return self.restart_ok

    def probe(self, port, model):
        self.probes.append((port, model))
        return self.health.get(port, True)

    def write(self, i, failed=False, n=1):
        t = self.targets[i]
        for _ in range(n):
            k = self.n_written.setdefault(i, 0)
            self.n_written[i] = k + 1
            d = self.root / t.tag / "test" / f"task{k}" / "L1"
            d.mkdir(parents=True, exist_ok=True)
            f = d / "result.json"
            f.write_text(json.dumps({"agent_status": "timeout" if failed else "exit 0"}))
            os.utime(f, (self.clock.t, self.clock.t))

    def finish(self, code=0):
        for h in self.sup.handles:
            h.code = code

    def sleep(self, s):
        self.clock.sleep(s)
        if self.script:
            step = self.script.pop(0)
            if step:
                step(self)

    def run(self):
        return self.sup.run()


def ok_tick(sim):
    sim.write(0)
    sim.write(1)


def test_a_healthy_run_finishes_clean_and_logs_a_progress_line_every_minute(tmp_path):
    sim = Sim(tmp_path, script=[ok_tick, ok_tick, ok_tick, lambda s: s.finish(0)])
    out = sim.run()
    assert out.code == 0 and out.restarts == 0 and sim.launches == [False] and sim.restart_calls == []
    assert len(sim.logs) == 4                                  # one per tick
    line = sim.logs[2]
    assert "eval-L1" in line and "m0 3/20" in line and "m1 3/20" in line
    assert "trials/min" in line and "ETA" in line and "accrued $1.50" in line
    assert sim.sup.final_rate == pytest.approx(6 / 4.0)       # 6 results in 4 minutes


def test_no_new_result_for_five_minutes_stops_the_harnesses_restarts_the_servers_once_and_resumes(tmp_path):
    script = [ok_tick] + [None] * 5 + [ok_tick, ok_tick, lambda s: s.finish(0)]
    sim = Sim(tmp_path, script=script)
    out = sim.run()
    assert out.code == 0 and out.restarts == 1
    assert sim.launches == [False, True]                       # the second launch is --retry-failed
    assert sorted(sim.stopped) == [0, 1] and len(sim.restart_calls) == 1
    assert any("STALL: no new result.json for 5.0 min" in ln for ln in sim.logs)
    assert any("restarting the servers (recovery 1 of 1)" in ln for ln in sim.logs)


def test_the_second_stall_stops_the_stage_with_a_nonzero_exit_and_a_clear_message(tmp_path):
    sim = Sim(tmp_path, script=[None] * 30)
    out = sim.run()
    assert out.code == SUP.EXIT_STALLED != 0 and out.restarts == 1
    assert "STOPPED after 1 recovery attempt(s)" in out.message and "STILL BILLING" in out.message
    assert "STALL" in out.message and "vllm_*.log" in out.message
    assert len(sim.restart_calls) == 1 and sim.launches == [False, True]
    # bounded: first stall at 5 min, restart, second stall 5 min after it: about 10 ticks, not hours
    assert sim.clock.t - 1_000_000.0 <= 11 * 60


def test_a_stalled_run_burns_at_most_two_detection_windows(tmp_path):
    sim = Sim(tmp_path, script=[None] * 100)
    sim.run()
    assert sim.clock.t - 1_000_000.0 == pytest.approx(10 * 60.0)


def test_a_failed_server_restart_stops_the_stage_too(tmp_path):
    sim = Sim(tmp_path, script=[None] * 10)
    sim.restart_ok = False
    out = sim.run()
    assert out.code == SUP.EXIT_STALLED and "server restart failed" in out.message and sim.launches == [False]


def test_a_high_share_of_harness_failures_among_the_last_results_is_a_trigger_even_while_results_arrive(tmp_path):
    def fail_tick(sim):
        sim.write(0, failed=True, n=2)
        sim.write(1)
    sim = Sim(tmp_path, script=[fail_tick] * 6, window=10, error_share=0.5)
    out = sim.run()
    assert out.code == SUP.EXIT_STALLED
    assert any("m0: " in ln and "harness failures (threshold 50%)" in ln for ln in sim.logs)
    assert not any("m1: " in ln and "harness failures" in ln for ln in sim.logs)


def test_only_results_written_since_the_restart_count_toward_the_error_share(tmp_path):
    def burst(sim):
        sim.write(0, failed=True, n=6)
        sim.write(1)
    sim = Sim(tmp_path, script=[burst, ok_tick, ok_tick, ok_tick, lambda s: s.finish(0)])
    out = sim.run()
    # the six failures trigger the first recovery; after it, the old failures must not trigger again
    assert out.code == 0 and out.restarts == 1


def test_a_few_failures_below_the_threshold_do_not_trigger(tmp_path):
    def tick(sim):
        sim.write(0, failed=True)
        sim.write(0, n=3)
        sim.write(1)
    sim = Sim(tmp_path, script=[tick, tick, tick, lambda s: s.finish(0)])
    assert sim.run().code == 0


def test_two_failed_health_probes_in_a_row_stop_and_restart(tmp_path):
    def bad(sim):
        sim.health[8001] = False
        ok_tick(sim)
    sim = Sim(tmp_path, script=[bad, ok_tick, ok_tick, lambda s: (s.health.update({8001: True}), ok_tick(s)),
                                ok_tick, lambda s: s.finish(0)])
    out = sim.run()
    assert out.code == 0 and out.restarts == 1
    assert any("health probe failed 2 times in a row: m1 (:8001)" in ln for ln in sim.logs)
    assert (8000, "m0") in sim.probes and (8001, "m1") in sim.probes     # every port, every tick


def test_a_single_failed_health_probe_is_forgiven_when_the_next_one_answers(tmp_path):
    def blip(sim):
        sim.health[8000] = False
        ok_tick(sim)

    def recover(sim):
        sim.health[8000] = True
        ok_tick(sim)
    sim = Sim(tmp_path, script=[blip, recover, blip, recover, lambda s: s.finish(0)])
    out = sim.run()
    assert out.code == 0 and out.restarts == 0


def test_a_harness_process_that_exits_nonzero_is_a_trigger(tmp_path):
    sim = Sim(tmp_path, script=[ok_tick, lambda s: s.finish(2)] + [None] * 3 + [lambda s: s.finish(0)])
    out = sim.run()
    assert out.restarts == 1 and out.code == 0
    assert any("exited with status [2, 2]" in ln for ln in sim.logs)


def test_a_sweep_that_finished_with_mostly_failures_is_not_called_clean(tmp_path):
    def burst_and_finish(sim):
        sim.write(0, failed=True, n=8)
        sim.write(1, n=8)
        sim.finish(0)
    sim = Sim(tmp_path, script=[burst_and_finish, ok_tick, ok_tick, lambda s: s.finish(0)])
    out = sim.run()
    assert out.code == 0 and out.restarts == 1     # not accepted as finished: retried, then clean
    assert sim.launches == [False, True]


def test_an_exception_mid_run_still_stops_every_harness_process(tmp_path):
    def boom(sim):
        raise KeyboardInterrupt
    sim = Sim(tmp_path, script=[ok_tick, boom])
    with pytest.raises(KeyboardInterrupt):
        sim.run()
    assert sorted(sim.stopped) == [0, 1]


def test_a_supervised_target_is_read_off_its_own_command_line():
    cmds = P.gate_cmds(cfg())
    t = SUP.Target.from_cmd(cmds[1].argv, cmds[1].env)
    assert t == SUP.Target("amd-r-2b", "amd2-r", "test", "L1", 8004, 60)
    l1 = next(s for s in plan_for() if s.name == "eval-L1").cmds[0]
    assert SUP.Target.from_cmd(l1.argv, l1.env).expected == 250
    s2 = next(s for s in plan_for() if s.name == "eval-sample2").cmds[0]
    assert SUP.Target.from_cmd(s2.argv, s2.env).expected == 500       # two samples: 250 x 2


class RecProc:
    """SUP.Proc stand-in that records what the driver starts."""
    started: list = []

    def __init__(self, argv, env):
        self.argv, self.env, self.code = argv, env, None
        RecProc.started.append(self)

    def poll(self):
        return self.code

    def stop(self, grace=0):
        self.code = -15


def run_supervised_sim(tmp_path, monkeypatch, step_name, script):
    c = gate_ledger(tmp_path)
    c.stall_minutes, c.error_window, c.error_share = 5.0, 10, 0.5
    steps = plan_for(c)
    step = next(s for s in steps if s.name == step_name)
    RecProc.started = []
    monkeypatch.setattr(SUP, "Proc", RecProc)
    targets = [SUP.Target.from_cmd(cmd.argv, cmd.env) for cmd in step.cmds]
    clock, runs = Clock(1_000_000.0), tmp_path / "runs"
    restarted = []

    def sleep(s):
        clock.sleep(s)
        script(clock, runs, targets, restarted)
    code = driver.run_supervised(c, step, steps, runs_root=runs, sleep=sleep, clock=clock,
                                 probe=lambda port, model: True, restart=lambda: restarted.append(1) or True)
    return c, code, restarted


def write_results(runs, t, n, clock, status="exit 0"):
    for k in range(n):
        d = runs / t.tag / "test" / f"task{k}" / "L1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps({"agent_status": status}))
        os.utime(d / "result.json", (clock.t, clock.t))


def test_the_driver_resumes_with_retry_failed_after_a_recovery_and_not_before(tmp_path, monkeypatch):
    ticks = []

    def script(clock, runs, targets, restarted):
        ticks.append(1)
        if len(ticks) == 1:
            for t in targets:
                write_results(runs, t, 3, clock)
        elif len(ticks) >= 8:
            for p in RecProc.started:
                p.code = 0
    c, code, restarted = run_supervised_sim(tmp_path, monkeypatch, "eval-L1", script)
    assert code == 0 and restarted == [1]
    first, second = RecProc.started[:5], RecProc.started[5:]
    assert len(first) == len(second) == 5
    assert not any("--retry-failed" in p.argv for p in first)
    assert all("--retry-failed" in p.argv for p in second)
    assert all("SMOL_LADDER_BASE_URL" in p.env and "DIGITALOCEAN_ACCESS_TOKEN" not in p.env for p in first)
    assert (Path(c.local_logs) / "progress-eval-L1.log").read_text().count("\n") >= 8


def test_a_second_stall_in_the_driver_writes_the_ledger_a_note_and_returns_nonzero(tmp_path, monkeypatch):
    c, code, restarted = run_supervised_sim(tmp_path, monkeypatch, "eval-L1", lambda *a: None)
    assert code == SUP.EXIT_STALLED and restarted == [1]
    notes = [e for e in L.read(Path(c.ledger)) if e["event"] == L.NOTE]
    assert notes and "STOPPED after 1 recovery attempt(s)" in notes[-1]["text"]


def test_a_finished_gate_eval_records_the_measured_trials_per_minute(tmp_path, monkeypatch):
    ticks = []

    def script(clock, runs, targets, restarted):
        ticks.append(1)
        for t in targets:
            write_results(runs, t, len(ticks) * 5, clock)
        if len(ticks) == 6:
            for p in RecProc.started:
                p.code = 0
    c, code, _ = run_supervised_sim(tmp_path, monkeypatch, "gate-eval", script)
    assert code == 0
    last = [e for e in L.read(Path(c.ledger)) if "trials_per_min" in e][-1]
    assert last["trials_per_min"] == pytest.approx(60 / 6.0, abs=0.01) and last["droplet_id"] == DROPLET


def test_restart_servers_reruns_the_matching_serve_step_records_it_and_reopens_the_tunnel(tmp_path):
    c = gate_ledger(tmp_path)
    steps = plan_for(c)
    gate_serve = next(s for s in steps if s.name == "gate-serve")
    calls = []

    def runner(cmd, host, capture=False, timeout=None):
        calls.append((cmd, capture, timeout))
        return 0, "ADAPTER_CHECK model=amd-r-2b differs=1 tool_calls_ok=1\n"
    assert driver.restart_servers(c, gate_serve, runner=runner, tunnel=lambda cfg: 0) is True
    assert calls[0][0] == gate_serve.cmds[0] and calls[0][1] is True
    assert L.read(Path(c.ledger))[-1]["serve_gate"] is True
    assert driver.restart_servers(c, gate_serve, runner=lambda *a, **k: (1, ""), tunnel=lambda cfg: 0) is False
    assert driver.restart_servers(c, gate_serve, runner=runner, tunnel=lambda cfg: 1) is False


def test_eval_steps_run_supervised_and_never_through_the_unsupervised_runner():
    src = inspect_source(driver.run_steps)
    assert 'step.phase == "eval" or step.name == "gate-eval"' in src and "run_supervised(" in src
    gate_branch = src[src.index('step.phase == "eval" or step.name == "gate-eval"'):]
    assert gate_branch.index("run_supervised(") < gate_branch.index("run_parallel(")


def test_the_supervisor_stops_by_process_group_of_its_own_child_never_by_name():
    src = Path(SUP.__file__).read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("#", '"""')))
    assert "pkill" not in code and "pgrep" not in code and "killall" not in code
    assert "start_new_session=True" in code and "os.killpg(pgid" in code and "os.getpgrp()" in code


def test_stopping_a_proc_kills_its_whole_tree_and_leaves_the_parent_alive(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    proc = SUP.Proc(["bash", "-c", f"sleep 300 & echo $! > {pidfile}; wait"], dict(os.environ))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        subprocess.run(["sleep", "0.05"])
    grandchild = int(pidfile.read_text())
    assert os.getpgid(proc.pid) == proc.pid != os.getpgrp()      # its own group: ours is untouched
    proc.stop(grace=5)
    assert proc.poll() is not None
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        subprocess.run(["sleep", "0.05"])
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
    os.kill(os.getpid(), 0)                                       # and so is this process


def test_the_health_probe_needs_a_real_completion_and_times_out_on_a_hung_engine():
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        mode = "ok"

        def do_POST(self):
            if H.mode == "hang":
                subprocess.run(["sleep", "2"])
            body = json.dumps({"choices": [{"message": {"content": "x"}}]} if H.mode != "empty" else {}).encode()
            self.send_response(200 if H.mode != "500" else 500)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert SUP.probe_http(port, "m", timeout=1.0) is True
        H.mode = "empty"
        assert SUP.probe_http(port, "m", timeout=1.0) is False
        H.mode = "500"
        assert SUP.probe_http(port, "m", timeout=1.0) is False
        H.mode = "hang"
        assert SUP.probe_http(port, "m", timeout=0.3) is False
    finally:
        srv.shutdown()
    assert SUP.probe_http(port, "m", timeout=0.3) is False        # nothing listening any more


def test_supervision_defaults_are_five_minutes_ten_results_half_failing_and_two_probe_failures():
    assert (SUP.DEFAULT_STALL_MIN, SUP.DEFAULT_WINDOW, SUP.DEFAULT_ERROR_SHARE, SUP.HEALTH_FAILS) == (5.0, 10, 0.5, 2)
    assert SUP.MIN_TICK_S == 60.0 and SUP.HEALTH_TIMEOUT_S <= 30.0
    c = P.Config()
    assert (c.stall_minutes, c.error_window, c.error_share) == (5.0, 10, 0.5)


# ═══ concurrency, flags and robustness ═══════════════════════════════════════════════════════════

def test_the_default_concurrency_per_model_is_the_measured_safe_8():
    assert P.Config().workers == 8 and P.MAX_WORKERS_SAFE == 10


def test_the_driver_refuses_more_than_ten_workers_per_model_unless_i_know_is_passed(tmp_path):
    for n in ("11", "20"):
        out = run_driver(tmp_path, "plan", "--workers", n)
        assert out.returncode != 0 and "above 10" in out.stderr and "--i-know" in out.stderr
        assert "20 per model" in out.stderr
    assert run_driver(tmp_path, "plan", "--workers", "10").returncode == 0
    ok = run_driver(tmp_path, "dry-run", "--workers", "12", "--i-know")
    assert ok.returncode == 0 and "--workers 12" in ok.stdout
    assert run_driver(tmp_path, "plan", "--workers", "0").returncode != 0


def test_the_eval_only_list_and_checkpoint_flags_reach_the_plan(tmp_path):
    out = run_driver(tmp_path, "dry-run", "--eval-only", "R=AdithyaSK/smoldataenvs-sft-2b-v0,X=a/b",
                     "--ckpt-steps", "20", "--train-attempts", "5")
    assert "X=a/b:8005" in out.stdout and "--save-steps 20 --max-attempts 5" in out.stdout
    assert "amd-x-2b" in out.stdout
    bad = run_driver(tmp_path, "plan", "--eval-only", "X=a/b")
    assert bad.returncode != 0 and "gate" in bad.stderr
    assert run_driver(tmp_path, "plan", "--eval-only", "nonsense").returncode != 0


def test_the_sync_pull_names_the_remote_entries_instead_of_a_bare_dot_that_scp_rejects():
    pull = next(s for s in plan_for() if s.name == "sync-pull").cmds[0]
    remote = pull.argv[-2]
    assert remote.endswith("/var/log/smol-ladder/*") and not remote.endswith(".")
    assert pull.argv[-1] == "logs/amd/" and "-r" in pull.argv
    assert "unexpected filename" in Path(driver.__file__).read_text() or "refuses a bare" in next(
        s for s in plan_for() if s.name == "sync-pull").note
    assert "/." not in remote


def agent_env(sock):
    return {"SSH_AUTH_SOCK": sock} if sock is not None else {}


def test_the_ssh_agent_socket_comes_from_the_environment_when_it_exists():
    env = agent_env("/tmp/agent.sock")
    assert driver.ensure_ssh_agent(env, 1000, exists=lambda p: p == "/tmp/agent.sock") == (
        "/tmp/agent.sock", "SSH_AUTH_SOCK from the environment")
    assert env["SSH_AUTH_SOCK"] == "/tmp/agent.sock"


def test_a_missing_or_dead_agent_socket_falls_back_to_the_keyring_socket_and_exports_it():
    fb = "/run/user/1000/keyring/ssh"
    for env in (agent_env(None), agent_env("/tmp/ssh-gone/agent.1")):
        sock, why = driver.ensure_ssh_agent(env, 1000, exists=lambda p: p == fb)
        assert sock == fb and env["SSH_AUTH_SOCK"] == fb and why.startswith("fallback")
    assert "gone" in driver.ensure_ssh_agent(agent_env("/x"), 1000, exists=lambda p: p == fb)[1]
    assert "not set" in driver.ensure_ssh_agent(agent_env(None), 1000, exists=lambda p: p == fb)[1]


def test_with_no_agent_at_all_nothing_is_exported_and_the_status_says_so():
    env = agent_env(None)
    sock, why = driver.ensure_ssh_agent(env, 1000, exists=lambda p: False)
    assert sock == "" and "SSH_AUTH_SOCK" not in env and why.startswith("none")


def test_the_fallback_uses_the_callers_uid():
    assert driver.AGENT_FALLBACK.format(uid=1234) == "/run/user/1234/keyring/ssh"
    assert "/run/user/1000/keyring/ssh" == driver.AGENT_FALLBACK.format(uid=1000)


def test_the_key_state_reads_ssh_add_exit_codes():
    def runner(rc, out=""):
        return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=out, stderr="")
    assert driver.agent_key_state(runner(0, "256 SHA256:x me (ED25519)\n")) == "1 key(s) loaded"
    assert driver.agent_key_state(runner(1)) == "agent reachable but NO key loaded"
    assert driver.agent_key_state(runner(2)) == "agent unreachable"

    def missing(*a, **k):
        raise FileNotFoundError
    assert driver.agent_key_state(missing) == "could not run ssh-add"


def test_children_inherit_the_agent_socket_the_driver_chose(monkeypatch):
    from ops.amd import doapi
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/user/1000/keyring/ssh")
    assert doapi.child_env()["SSH_AUTH_SOCK"] == "/run/user/1000/keyring/ssh"
    assert "ensure_ssh_agent()" in inspect_source(driver.main)


def test_status_prints_the_agent_in_use_and_warns_that_a_relogin_kills_the_deadman(tmp_path, capsys):
    now = __import__("time").time()
    c = cfg(ledger=str(tmp_path / "l.jsonl"))
    driver.print_status(c, L.summarise([], now, 2.46), now, agent="/run/user/1000/keyring/ssh (fallback); 1 key(s) loaded")
    out = capsys.readouterr().out
    assert "ssh agent  /run/user/1000/keyring/ssh (fallback); 1 key(s) loaded" in out
    assert "NO LIVE DEADMAN" in out and "re-login kills the dead-man process" in out
    hb(tmp_path, now)
    driver.print_status(c, L.summarise([], now, 2.46), now, agent="")
    fresh = capsys.readouterr().out
    assert "HEARTBEAT FRESH" in fresh and "re-login" not in fresh and "ssh agent" not in fresh


def test_the_status_command_reports_the_agent_it_found(tmp_path):
    out = run_driver(tmp_path, "status")
    assert out.returncode == 0 and "ssh agent" in out.stdout


# ═══ the staged token counts must be the ones the plan is costed from ═════════════════════════════

def test_stage_computes_ab_as_a_plus_b_from_arm_b_v2(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "ja3_sft_v2.jsonl"
    a.write_text('{"messages": [{"role": "user", "content": "aaaa"}]}\n' * 3)
    b.write_text('{"messages": [{"role": "user", "content": "bb"}]}\n' * 2)
    counts = stage.build_token_counts([a], b, 8192, encode=lambda t: len(t))
    s = counts["sets"]
    assert s["AB"]["trained_tokens"] == s["A"]["trained_tokens"] + s["B"]["trained_tokens"]
    assert s["AB"]["rows"] == 5 and s["B"]["rows"] == 2 and s["AB"]["method"] == "A + B"
    assert "ja3_sft_v2.jsonl" in inspect_source(stage.stage) and "ja3_sft.jsonl" not in inspect_source(stage.stage).replace("ja3_sft.jsonl is v1", "")


def test_the_recounted_tokens_the_plan_falls_back_to_match_the_session_prep_numbers():
    assert P.RECOUNTED_TOKENS[8192] == {"A": (4439, 9_085_233), "B": (1122, 3_123_047)}
    assert 9_085_233 + 3_123_047 == 12_208_280


def test_the_stage_requires_the_new_scripts_in_the_pinned_commit():
    for f in ("ops/amd/sft_run.py", "ops/amd/merge_adapter.py", "ops/amd/probe_tools.py"):
        assert f in stage.REQUIRED_IN_COMMIT


# ═══ the runbook states the gate, the stall policy and the lessons, with the code's numbers ══════

def test_the_runbook_states_session_1s_measurements_as_measured_on_that_hardware():
    text = runbook()
    for needle in ("5,446", "47 s", "22 trials per minute", "57 minutes", "12 minutes",
                   "9,085,233", "3,123,047", "12,208,280", "measured in session 1"):
        assert needle in text, needle
    assert P.MEASURED_TOKENS_PER_S == 5446.0 and P.MEASURED_TRIALS_PER_MIN == 22.0
    assert round(5 * 250 / P.MEASURED_TRIALS_PER_MIN) == 57
    assert (P.RECOUNTED_TOKENS[8192]["A"][1], P.RECOUNTED_TOKENS[8192]["B"][1]) == (9_085_233, 3_123_047)


def test_the_runbook_explains_the_gate_its_verdicts_and_what_accept_gate_may_and_may_not_do():
    text = runbook()
    for needle in ("2a. The gate", "first 60 tasks", "L1 tags", "GO", "NO-GO", "STOP", "--accept-gate",
                   "paired comparison", "stop-reason histograms", "answer_submitted", "max_turns",
                   "context_exhausted", "cannot override", "stop-gate-server", "refuse without it"):
        assert needle in text, needle
    assert P.GATE_TASKS == 60 and P.Config().gate_margin == 0.05 and P.Config().gate_max_failures == 3
    assert "0.05" in text and "3 of 60" in text


def test_the_runbook_states_the_stall_policy_with_the_numbers_the_code_uses():
    text = runbook()
    assert "5 minutes" in text and "50%" in text and "last **10**" in text and "twice in a row" in text
    assert (SUP.DEFAULT_STALL_MIN, SUP.DEFAULT_ERROR_SHARE, SUP.DEFAULT_WINDOW, SUP.HEALTH_FAILS) == (5.0, 0.5, 10, 2)
    for needle in ("--retry-failed", "restarted once", "exits non-zero (75)", "by the process group",
                   "never by\npattern-matching", "progress-<step>.log", "trials per\nminute"):
        assert needle in text, needle
    assert SUP.EXIT_STALLED == 75
    assert "--workers" in text and "--i-know" in text and "default **8**" in text and "above 10" in text


def test_the_runbook_says_to_restart_the_deadman_after_a_relogin_and_names_the_agent_fallback():
    text = runbook()
    assert "re-login" in text and "stale\nheartbeat" in text and "NO LIVE DEADMAN" in text
    assert "/run/user/<uid>/keyring/ssh" in text and "SSH_AUTH_SOCK" in text
    assert "Restart the dead-man" in text or "restart the dead-man" in text.lower()


def test_the_runbook_lists_the_lessons_from_session_1_each_with_its_guard():
    text = runbook()
    block = text[text.index("## 11. Lessons from session 1"):text.index("## Sources")]
    items = re.findall(r"(?m)^\d+\. \*\*", block)
    assert len(items) == 9
    for needle in ("two hours", "byte-identical", "released upstream adapter", "Engine core initialization failed",
                   "device wedged", "Please wait while we get your droplet ready", "unexpected filename",
                   "dead-man process", "LoRA mode does not work", "0.15-0.2", "180-200", "amd2"):
        assert needle in block, needle


def test_the_runbook_says_why_the_fast_kernels_are_not_installed_and_how_to_try_them():
    text = runbook()
    assert "causal_conv1d" in text and "flash-linear-attention" in text and "Not installed, deliberately" in text
    assert "constraints.txt flash-linear-attention" in text


def test_the_runbook_defaults_match_the_code():
    text = runbook()
    c = P.Config()
    assert (c.ckpt_steps, c.train_attempts, c.workers, c.samples, c.limit) == (50, 3, 8, 1, 250)
    assert "`--ckpt-steps` (default 50)" in text and "`--train-attempts` (default 3)" in text
    assert "0.17 of the GPU" in text and "ports 8000-8004" in text or "8000-8004" in text
    assert "eval --stage sample2" in text and "--eval-only" in text


# ═══ serve.sh: only finished adapters, only merges made from them, one engine at a time ═══

def test_a_trained_arms_adapter_on_disk_without_done_is_a_checkpoint_and_is_not_served(serve_env):
    (serve_env["root"] / "runs" / "sft_a" / ".done").unlink()      # mid-training: files copied, no marker
    out = run_serve(serve_env, "--wait", "--arms", "A", "--hub", HUB_R, check=False)
    assert out.returncode != 0 and "no .done" in out.stderr
    assert ("8001", "amd-a-2b") not in starts(serve_env)


def test_the_hub_fallback_needs_the_final_marker_in_the_downloaded_repo(serve_env):
    (serve_env["root"] / "runs" / "sft_a" / ".done").unlink()
    hub_dir = serve_env["tmp"] / "hub_r"                          # what the stub download returns
    out = run_serve(serve_env, "--wait", "--arms", "A", extra_env={}, check=False)
    assert out.returncode != 0 and "no final.done" in out.stderr
    (hub_dir / "final.done").write_text("done\n")
    run_serve(serve_env, "--wait", "--arms", "A")
    assert ("8001", "amd-a-2b") in starts(serve_env)


def test_a_merge_made_from_an_older_adapter_is_redone_not_served(serve_env):
    import test_merge_adapter as TM
    run_serve(serve_env, "--wait", "--arms", "A")
    merged = serve_env["root"] / "runs" / "merged_amd-a-2b"
    before = json.loads((merged / "merge_report.json").read_text())["adapter_sha256"]
    base = TM.make_base(serve_env["tmp"] / "tinybase2")
    shutil.rmtree(serve_env["root"] / "runs" / "sft_a")
    TM.make_adapter(serve_env["root"] / "runs" / "sft_a", base, TM.TARGETS)   # "retrained"
    (serve_env["root"] / "runs" / "sft_a" / ".done").write_text("done\n")
    run_serve(serve_env, "--wait", "--arms", "A")
    after = json.loads((merged / "merge_report.json").read_text())["adapter_sha256"]
    assert before != after


def test_the_serve_script_starts_an_engine_only_after_the_previous_one_is_up():
    text = (OPS / "serve.sh").read_text()
    loop = text[text.index('for i in "${!M_NAMES[@]}"; do\n  start_model'):]
    assert loop.index("start_model") < loop.index("bring_up") < loop.index("done")
    assert 'for i in "${!M_NAMES[@]}"; do bring_up' not in text


def test_the_kv_budget_is_equal_explicit_and_can_fall_back_to_equal_gpu_shares(serve_env):
    rec = record_argv(serve_env)
    run_serve(serve_env, "--arms", "A", "--hub", HUB_R, extra_env={"AMD_KV_CACHE_GIB": "0", "AMD_PREFIX_ARGS": ""})
    argv = rec.read_text().splitlines()
    assert argv.count("--gpu-memory-utilization") >= 1 and "--kv-cache-memory-bytes" not in argv
    assert "--enable-prefix-caching" not in argv


def test_an_engine_that_fails_naming_prefix_caching_is_restarted_without_it(serve_env):
    stub = Path((serve_env["root"] / ".syspy").read_text().strip())
    log = serve_env["tmp"] / "log"
    stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        port=""; name=""; prefix=0
        while [[ $# -gt 0 ]]; do
          case "$1" in --port) port="$2";; --served-model-name) name="$2";; --enable-prefix-caching) prefix=1;; esac; shift
        done
        echo "$port $name prefix=$prefix" >> "{serve_env['started']}"
        if [[ "$prefix" == 1 && "$port" == 8000 ]]; then echo "mamba cache mode align unsupported" >&2; exit 1; fi
        echo "$name" > "{serve_env['tmp']}/stubs/up.$port"
        exec sleep 6
        """))
    out = run_serve(serve_env, "--wait", "--arms", "", "--hub", HUB_R)
    assert "WITHOUT it" in out.stderr
    lines = serve_env["started"].read_text().splitlines()
    assert [ln for ln in lines if ln.startswith("8000")] == ["8000 amd-base-2b prefix=1", "8000 amd-base-2b prefix=0"]


def test_engines_are_started_in_order_and_the_next_one_waits_for_the_previous_ready_line(serve_env):
    out = run_serve(serve_env, "--wait", "--arms", "A,B", "--hub", HUB_R)
    assert [t[0] for t in starts(serve_env)] == ["8000", "8001", "8002", "8004"]
    assert out.stdout.count("READY_AFTER_S=") == 4



# ═══ H2: names carry the session; .done only after a verified upload; verify compares sha256 ═══

class FinalApi:
    """A stub of the HfApi calls `resume.finalize` makes. `corrupt` makes the Hub report another sha."""

    def __init__(self, corrupt=False):
        self.uploads, self.corrupt = [], corrupt

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        self.uploads.append(path_in_repo)
        if path_in_repo == resume.ADAPTER:
            self.sha = stage.sha256_file(Path(path_or_fileobj))

    def get_paths_info(self, repo, paths, repo_type="model"):
        sha = "0" * 64 if self.corrupt else self.sha
        return [type("F", (), {"lfs": type("L", (), {"sha256": sha})()})()]


def finished_arm_dir(tmp_path) -> Path:
    out = tmp_path / "sft_a"
    out.mkdir()
    write_safetensors(out / resume.ADAPTER)
    (out / "adapter_config.json").write_text("{}")
    return out


def test_done_is_written_only_after_the_upload_was_verified_and_the_marker_goes_up_last(tmp_path):
    out, api = finished_arm_dir(tmp_path), FinalApi()
    resume.finalize(out, "ns/sft-a-s2", api)
    assert api.uploads == [resume.ADAPTER, "adapter_config.json", resume.HUB_DONE]
    assert (out / resume.DONE).exists()


def test_a_hub_copy_that_does_not_match_leaves_no_done_and_no_final_marker(tmp_path):
    out, api = finished_arm_dir(tmp_path), FinalApi(corrupt=True)
    with pytest.raises(SystemExit, match="does not match"):
        resume.finalize(out, "ns/sft-a-s2", api)
    assert not (out / resume.DONE).exists() and resume.HUB_DONE not in api.uploads


def test_a_truncated_adapter_is_never_uploaded_or_marked_done(tmp_path):
    out, api = finished_arm_dir(tmp_path), FinalApi()
    write_safetensors(out / resume.ADAPTER, truncate=4)
    with pytest.raises(SystemExit):
        resume.finalize(out, "ns/sft-a-s2", api)
    assert api.uploads == [] and not (out / resume.DONE).exists()


def test_hub_repo_names_carry_the_session_and_are_the_same_in_python_and_shell(tmp_path):
    assert [P.hub_name(k) for k in ("A", "B", "AB", "artifacts")] == [
        "smol-ladder-sft-a-s2", "smol-ladder-sft-b-s2", "smol-ladder-sft-ab-s2", "smol-ladder-runs-s2"]
    for session in ("s2", "s3"):
        sh = subprocess.run(["bash", "-c", f'source "{OPS}/common.sh"; for k in A B AB artifacts; do amd_hub_name $k; done'],
                            env={**os.environ, "AMD_SESSION": session}, capture_output=True, text=True)
        assert sh.stdout.split() == [P.hub_name(k, session) for k in ("A", "B", "AB", "artifacts")]
    assert not any(n in (OPS / "common.sh").read_text() for n in ('AMD_HUB_ADAPTER_A="', 'AMD_HUB_ARTIFACTS="'))


def test_the_session_variable_reaches_the_droplet_through_remote_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("AMD_SESSION", "s7")
    stage.write_remote_env(tmp_path, "ns")
    assert "AMD_SESSION=s7" in (tmp_path / "remote.env").read_text()
    monkeypatch.delenv("AMD_SESSION")
    stage.write_remote_env(tmp_path, "ns")
    assert "AMD_SESSION=s2" in (tmp_path / "remote.env").read_text()


def test_driver_and_setup_use_the_suffixed_names():
    c = cfg()
    assert driver.hub_repos(c, "ns")["A"] == "ns/smol-ladder-sft-a-s2"
    setup = (OPS / "container_setup.sh").read_text()
    assert "smol-ladder-sft-a-{session}" in setup and "smol-ladder-runs-{session}" in setup


class VerifyApi:
    def __init__(self, files, sha, final_sha=None):
        self.files, self.sha, self.final_sha = files, sha, final_sha

    def list_repo_files(self, repo): return list(self.files)

    def get_paths_info(self, repo, paths, repo_type="model"):
        return [type("F", (), {"lfs": type("L", (), {"sha256": self.sha})()})()]


def test_push_artifacts_verify_needs_the_final_marker_and_the_local_adapters_sha(tmp_path):
    from ops.amd import push_artifacts as PA
    f = tmp_path / "adapter_model.safetensors"
    write_safetensors(f)
    sha = stage.sha256_file(f)
    ok = VerifyApi(["adapter_config.json", "adapter_model.safetensors", resume.HUB_DONE], sha)
    assert PA.verify(ok, {"ns/a": f}) == []
    no_marker = VerifyApi(["adapter_config.json", "adapter_model.safetensors"], sha)
    assert PA.verify(no_marker, {"ns/a": f}) == ["ns/a"]
    stale = VerifyApi(["adapter_config.json", "adapter_model.safetensors", resume.HUB_DONE], "1" * 64)
    assert PA.verify(stale, {"ns/a": f}) == ["ns/a"]


def test_verify_sync_compares_the_hub_adapter_with_the_local_final_adapter(tmp_path):
    c = cfg(local_logs=str(tmp_path / "logs"), limit=2, samples=1, arms=("A",))
    f = tmp_path / "logs" / "adapters" / "A" / "adapter_model.safetensors"
    f.parent.mkdir(parents=True)
    write_safetensors(f)
    (tmp_path / "logs" / "SHA256SUMS.artifacts").write_text(
        f"{stage.sha256_file(f)}  adapters/A/adapter_model.safetensors\n")
    make_runs(tmp_path / "runs", P.expected_trials(c))
    files = ["adapter_config.json", "adapter_model.safetensors", resume.HUB_DONE]
    good = lambda repo: files  # noqa: E731
    kw = dict(runs_root=tmp_path / "runs", namespace="ns")
    assert driver.verify_sync(c, hub_files=good, hub_sha256=lambda repo, name: stage.sha256_file(f), **kw) is True
    assert driver.verify_sync(c, hub_files=good, hub_sha256=lambda repo, name: "2" * 64, **kw) is False
    assert driver.verify_sync(c, hub_files=lambda r: files[:2], hub_sha256=lambda repo, name: stage.sha256_file(f), **kw) is False
