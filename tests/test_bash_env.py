"""The interpreter the bash protocol's commands run under.

The SFT rows were recorded in a Python 3.12 container whose `pip list` is in the data
(row 1852 of data/train/sft_upstream/train.jsonl: pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0 ...,
no pyarrow). The sandbox runs the model's commands under a dedicated environment built to that list,
and says in every result.json which interpreter it was and whether it fell back to the repo's venv.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from smol_ladder import bash_env as B


def fake_env(root: Path, pyver="3.12.12", packages=None) -> Path:
    """A directory shaped like a uv venv: pyvenv.cfg, bin/python, lib/python3.12/site-packages."""
    packages = packages or {"pandas": "3.0.3", "numpy": "2.4.6", "scikit_learn": "1.9.0"}
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").symlink_to(sys.executable)
    (root / "pyvenv.cfg").write_text(f"home = /x/bin\nversion_info = {pyver}\n")
    site = root / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    for name, ver in packages.items():
        (site / name).mkdir()
        (site / f"{name}-{ver}.dist-info").mkdir()
    return root


def test_the_default_is_the_dedicated_environment_when_it_exists(tmp_path):
    env = fake_env(tmp_path / "env")
    sb = B.resolve({}, default=env)
    assert sb.record["env"] == str(env) and sb.record["fallback"] is False
    assert sb.record["python"] == "3.12.12"
    assert sb.record["packages"]["pandas"] == "3.0.3"
    assert sb.record["packages"]["scikit-learn"] == "1.9.0"
    assert sb.version == "3.12" and sb.sites == (env / "lib/python3.12/site-packages",)


def test_a_missing_default_falls_back_to_the_current_venv_loudly(tmp_path, capsys):
    sb = B.resolve({}, default=tmp_path / "nope")
    assert sb.record["fallback"] is True
    assert "NOT the recording environment" in sb.record["warning"]
    assert sb.record["env"] == "current venv"
    assert "WARNING" in capsys.readouterr().err


def test_the_variable_overrides_the_default_and_a_bad_path_falls_back(tmp_path, capsys):
    env = fake_env(tmp_path / "env")
    assert B.resolve({B.ENV_VAR: str(env)}, default=tmp_path / "x").record["env"] == str(env)
    sb = B.resolve({B.ENV_VAR: str(tmp_path / "typo")}, default=env)
    assert sb.record["fallback"] is True and "typo" in sb.record["warning"]
    assert "WARNING" in capsys.readouterr().err


def test_current_is_an_explicit_choice_not_a_fallback(tmp_path, capsys):
    sb = B.resolve({B.ENV_VAR: "current"}, default=fake_env(tmp_path / "env"))
    assert sb.record["env"] == "current venv" and sb.record["fallback"] is False
    assert "warning" not in sb.record and capsys.readouterr().err == ""


def test_the_record_names_the_key_packages_it_finds(tmp_path):
    env = fake_env(tmp_path / "env", packages={"pandas": "3.0.3", "pyarrow": "9.9", "junk": "1"})
    rec = B.resolve({}, default=env).record
    assert rec["packages"] == {"pandas": "3.0.3", "pyarrow": "9.9"}


def test_jail_args_bind_the_environments_site_packages_under_its_own_python_version(
        tmp_path, monkeypatch):
    import smol_ladder.run_ladder as runner
    env = fake_env(tmp_path / "env")
    monkeypatch.setenv(B.ENV_VAR, str(env))
    B.cached.cache_clear()
    inputs = tmp_path / "in"; inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    for d in ("w", "h", "e"):
        (tmp_path / d).mkdir()
    args, python, envv, _ = runner.jail_bash(inputs, tmp_path / "w", tmp_path / "h", tmp_path / "e")
    i = args.index(str(env / "lib/python3.12/site-packages"))
    assert args[i - 1] == "--ro-bind" and args[i + 1] == "/usr/local/lib/python3.12/site-packages"
    assert python == "/usr/local/bin/python3"
    B.cached.cache_clear()


def test_every_bash_result_records_the_sandbox_interpreter(tmp_path, monkeypatch):
    if shutil.which("bwrap") is None:
        pytest.skip("needs bubblewrap")
    from tests.test_bash_sandbox import run_commands
    monkeypatch.setenv(B.ENV_VAR, "current")
    B.cached.cache_clear()
    result, _ = run_commands(tmp_path, monkeypatch, ['echo -n "42" > /workdir/answer.txt'])
    rec = result["bash_sandbox"]
    assert rec["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert "pandas" in rec["packages"]
    json.dumps(result)
    B.cached.cache_clear()


DEDICATED = B.DEFAULT_ENV


@pytest.mark.skipif(not (DEDICATED / "bin" / "python").exists() or shutil.which("bwrap") is None,
                    reason="the dedicated evaluation environment is not built here")
def test_the_real_dedicated_environment_runs_python_312_with_the_recorded_versions(tmp_path, monkeypatch):
    from tests.test_bash_sandbox import run_commands
    monkeypatch.delenv(B.ENV_VAR, raising=False)
    B.cached.cache_clear()
    result, out = run_commands(tmp_path, monkeypatch, [
        "python3 -c \"import sys,pandas,numpy,sklearn;print(sys.version_info[:2],pandas.__version__,"
        "numpy.__version__,sklearn.__version__)\"",
        "python3 -c \"import pyarrow\" 2>&1 | tail -1",
        'echo -n "42" > /workdir/answer.txt'])
    assert out[0] == "(3, 12) 3.0.3 2.4.6 1.9.0\n"
    assert "No module named 'pyarrow'" in out[1]
    assert result["bash_sandbox"]["fallback"] is False
    B.cached.cache_clear()
