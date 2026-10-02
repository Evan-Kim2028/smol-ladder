"""The two protocols the released 2B models were trained and evaluated under, verbatim.

`smoldataenvs-grpo-2b-v0` and `smoldataenvs-sft-2b-v0` are not agent models in our sense. Their
training data fixes the prompt, the tool schema, the answer protocol and the number of turns,
and every one of those is a protocol, not a style:

- **GRPO** (`scripts/rollout.py`, and the only thing `eval_pass1.py` ever scores) is *one turn*.
  No tools at all. The model writes a Python program in a ```python fence, the sandbox runs it,
  and the last non-empty line it printed is the answer. `enable_thinking=False` on the chat
  template, `MAX_NEW_TOKENS=1024`, greedy.
- **SFT** (`FineEnvs/SmolDataEnvs-sft`, 4,677 trajectories) is a *bash* agent: one tool named
  `bash`, non-stateful between calls, and the answer submitted by
  `echo -n "<value>" > /workdir/answer.txt`. Three to twelve turns in the published rows.

Neither is our loop. Ours offers `run_shell` + `write_solution`, asks for a persistent
`solution.py`, and re-runs that file offline before grading anything. A 2B model trained on the
one-turn protocol scores near zero through our loop and a 2B model trained on the bash protocol
scores near zero through theirs -- not because it is bad at the task, because we asked for a
different thing. That is why `run_ladder --agent` exists: the numbers people quote for these
models are protocol numbers, so reproducing them requires the protocol.

Everything here is copied from upstream, not paraphrased. `docs/LOCAL_MODELS.md` cites each
source and says what had to change.
"""

from __future__ import annotations

import re

# ── GRPO / eval_pass1.py: one program, no tools ───────────────────────────────

PROGRAM_SYSTEM = (
    "You are a data analyst. You answer questions about CSV files by writing a short "
    "Python program and reading what it prints."
)

PROGRAM_PROMPT = """{question}

The files are in /home/user/input and your program runs in that directory:
{files}

Write one Python program in a ```python block, then stop.

- Look at the data if you need to, then compute the answer.
- The LAST thing the program prints must be the answer on its own: a bare number
  (no commas or units), a short label, yes/no, or a comma-separated list.
- Keep it under 40 lines. pandas, numpy, scipy, sklearn and statsmodels are installed."""

# No file names: upstream says so explicitly rather than printing an empty list, because the
# data is still staged and the model needs to be told to go and look for it.
NO_FILES = (
    "- No file names were provided by the dataset. The data is still staged in "
    "/home/user/input; list /home/user/input first, for example with "
    "os.listdir('/home/user/input'), then read the discovered files."
)


def program_prompt(question: str, files: list[str]) -> list[dict]:
    """The one prompt shape `eval_pass1.py` scores under. Same function, same text."""
    listing = "\n".join(f"- {f}" for f in files) if files else NO_FILES
    return [
        {"role": "system", "content": PROGRAM_SYSTEM},
        {"role": "user", "content": PROGRAM_PROMPT.format(question=question, files=listing)},
    ]


# ── SFT: one bash tool, answer to a file ──────────────────────────────────────

BASH_SYSTEM = (
    "You are an autonomous data-analysis agent operating in a sandboxed Linux container. Your "
    "only tool is `bash`. The dataset files are in /home/user/input/. Python 3 + pandas + "
    "numpy + scikit-learn + scipy are pre-installed.\n\n"
    "Work the problem step-by-step: first inspect the data (ls, head, shape, dtypes), then plan, "
    "then compute, then submit.\n\n"
    "To submit your final answer you MUST call the `bash` tool to write it to "
    "/workdir/answer.txt, e.g. `echo -n \"<value>\" > /workdir/answer.txt`. Keep the answer "
    "short and concise. Do NOT end your turn without submitting."
)

# The user turn of every one of the 4,673 SmolDataEnvs-sft trajectories, reconstructed rather than
# copied: grouping the rows by their user message with the three variable parts (file list,
# question, answer-format line) cut out leaves exactly ONE skeleton, so this is the template and
# the rows are its instances. It is NOT hard-wrapped: an earlier copy of this constant wrapped its
# lines at ~100 characters and carried an extra blank line when the answer-format line was empty,
# and a model trained on the unwrapped text and prompted with the wrapped one is being asked a
# different question. `{answer_format}` is either "" or the per-task line plus its blank line --
# when it is empty the data has ONE blank line between "compute." and "Answer with", not two.
BASH_USER = (
    "You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect "
    "the files and compute the answer.\n\n"
    "Files (in /home/user/input, no subfolders):\n{files}\n\n"
    "Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, "
    "sqlite3, plotly (pip install more if needed).\n\n"
    "Question:\n{question}\n\n"
    "Work it out step by step \u2014 inspect the data first (head, shape, dtypes), then compute.\n\n"
    "{answer_format}"
    "Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short "
    "label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable "
    "answer, write: Not Applicable\n\n"
    "Write only that value to /workdir/answer.txt (e.g. `echo -n \"<value>\" > "
    "/workdir/answer.txt`), then stop."
)

DEFAULT_ANSWER_FORMAT = ""

# The per-task line sits between "compute." and "Answer with a single clean value". SmolDataEnvs
# rows carry it only inside their `instruction` column, so it is read back out of there.
_ANSWER_FORMAT = re.compile(r"\n\nWork it out step by step[^\n]*\n\n(.*?)Answer with a single clean value",
                            re.S)


def answer_format_of(row: dict) -> str:
    """The task's own answer-format line ("Express the value as a percentage ..."), or "".

    ~8.5% of SmolDataEnvs-sft rows have one. It is part of the question, not the protocol, so it
    comes from the task row (`instruction`) and rows without one -- every jupyter-agent and
    synthetic task -- get the empty string, which the template renders exactly as the data does.
    """
    m = _ANSWER_FORMAT.search(row.get("instruction") or "")
    return m.group(1).strip() if m else DEFAULT_ANSWER_FORMAT


BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the sandbox and return its combined stdout+stderr. The "
                "shell is non-stateful between calls (each call is a fresh `bash -c`). Use it to "
                "explore files (ls, head, cat), run Python (`python3 -c ...`), and write the "
                "final answer (`echo -n \"<value>\" > /workdir/answer.txt`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"],
            },
        },
    }
]

ANSWER_FILE = "answer.txt"

# Where the SFT trajectories' commands run, as the rows themselves show it: the tables at
# /home/user/input, the answer at /workdir/answer.txt, the shell's home /home/user (its `ls -la`
# in 3 rows), and a `/workdir` that is the only other directory the prompt names. The working
# directory is not recorded -- every one of the 17,099 commands uses absolute paths, or `cd`s first
# (6,174 of them to /home/user/input) -- so /workdir is a choice, not a measurement.
INPUT_DIR = "/home/user/input"
WORKDIR = "/workdir"
HOME_DIR = "/home/user"


def bash_prompt(question: str, files: list[str], answer_format: str = "") -> list[dict]:
    """The prompt each SmolDataEnvs-sft trajectory carries in its `user` turn.

    `answer_format` is the per-task line ("Answer as: <value>, <name> ...") that some rows have
    between the question and the generic format sentence (`answer_format_of(row)`). It is part of
    the question, not of the protocol; pass "" for the rows that have none. `files` is rendered as
    `- name` lines, one per file, in the order given.
    """
    listing = "\n".join(f"- {f}" for f in files)
    return [
        {"role": "system", "content": BASH_SYSTEM},
        {"role": "user", "content": BASH_USER.format(
            question=question, files=listing,
            answer_format=(answer_format + "\n\n") if answer_format else DEFAULT_ANSWER_FORMAT)},
    ]


# ── answer extraction, from rollout.py ────────────────────────────────────────

CODE_RE = re.compile(r"```(?:python3?|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(completion: str | list[dict]) -> str:
    """Last fenced block wins: models often think in one block and answer in the next."""
    if isinstance(completion, list):
        completion = completion[-1]["content"] if completion else ""
    blocks = CODE_RE.findall(completion or "")
    if blocks:
        return blocks[-1].strip()
    # no fence at all: treat the whole thing as code rather than scoring a zero for
    # punctuation. If it is prose it will fail to compile, which is its own signal.
    return (completion or "").strip()


def is_program(code: str) -> bool:
    """Does it parse? Upstream decides this here, before spending a sandbox on it."""
    try:
        compile(code, "<rollout>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


# Adapted from 04-data-agent/envs/whitebox-bash/grader.py, where 42% of partial
# credit once went to strings like `echo -n "2.14" > answer.txt`: the text contains
# the right number and grades as the right answer.
# A redirect only counts when something comes before it (`2.14 > answer.txt`): a value that
# starts with `>` is an answer, e.g. the gold answers `>50K`, `> 2 Years` and `>40hrs`.
# A pipe always counts: no gold answer contains one.
_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|\|{1,2}\s*\S+|\S\s*>{1,2}\s*\S+|\$\(|`",
)


def looks_like_a_command(answer: str) -> bool:
    """True when the 'answer' is really the line that would produce it."""
    return bool(answer) and bool(_COMMAND_SHAPED.search(answer.strip()))
