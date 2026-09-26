from sdt.grade import grade
from sdt.sandbox import run_script
from sdt.tasks import input_dir, load_split


def test_grader_gold_and_wrong():
    row = load_split("test")[0]
    assert grade(row, str(row["answer"])) == 1.0
    assert grade(row, "definitely wrong 12345") == 0.0


def test_sandbox_reads_input_offline(tmp_path):
    row = load_split("test")[0]
    inp = input_dir(row)
    s = tmp_path / "solution.py"
    s.write_text(
        "import os, socket\n"
        "print(sorted(os.listdir('input')))\n"
        "try:\n    socket.create_connection(('1.1.1.1', 80), 2); print('net')\n"
        "except OSError:\n    print('nonet')\n"
    )
    r = run_script(s, inp)
    assert r.returncode == 0, r.stderr
    assert row["files"][0] in r.stdout
    assert r.stdout.strip().endswith("nonet")
