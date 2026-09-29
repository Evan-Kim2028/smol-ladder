"""Shared fixtures. The `row` fixture is one task with a real table and a real reference, so a
test can build all five rungs without touching the Hub."""

import pandas as pd
import pytest

from smol_ladder import ladder as L

REFERENCE = (
    "import pandas as pd\n"
    "df = pd.read_csv('input/t.csv')\n"
    "x = df[df['flag'] == 1]\n"
    "print(x['col_a'].mean())\n"
)

ROW = {
    "task_id": "conformance_1",
    "question": "What is the mean of col_a for rows where flag is 1?",
    "files": ["t.csv"],
    "answer": "3.5",
    "reward_mode": "numeric",
    "atol": 1e-3,
    "rtol": 1e-3,
    "bucket_prefix": "conformance/one",
    "difficulty_tier": 1,
    "split": "test",
}


@pytest.fixture
def row(tmp_path, monkeypatch):
    """A task row with a table on disk and a verified reference, so every rung is buildable."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "t.csv").write_text("flag,col_a\n1,3.0\n1,4.0\n0,9.0\n")
    monkeypatch.setattr(L, "inputs_of", lambda split: (lambda _r: src))
    monkeypatch.setattr(L, "read_source", lambda row, split: REFERENCE)
    return dict(ROW)


@pytest.fixture
def frame():
    """A small table with a measure, a key, and junk columns. Team counts are deliberately
    unequal: a tied mode is rejected by design, so a tie would make mode tasks unverifiable."""
    return pd.DataFrame({
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "team": ["a", "b", "a", "b", "a", "b", "c", "c", "c", "c"] * 3,
        " id": range(30),
        "#": range(30),
    })
