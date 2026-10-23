"""Console approval gate: print the proposal and ask for an explicit y / yes.

Input, output and the terminal check are injected, so tests need no terminal. Anything
other than y / yes rejects, including no terminal, EOF and Ctrl-C. There is no re-prompt,
no timeout and no auto-approve.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

PROMPT = "Approve and submit these orders? [y/N] "
NO_TERMINAL = "No interactive terminal: rejected.\n"
NO_ANSWER = "\nNo answer: rejected.\n"
REJECTED = "Rejected.\n"

_YES = frozenset({"y", "yes"})


def _write_stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _stdin_is_interactive() -> bool:
    """Looks ``sys.stdin`` up at call time; a missing stdin is not interactive."""
    return sys.stdin is not None and sys.stdin.isatty()


class ConsoleApprovalGate:
    """ApprovalGate on a terminal: shows the summary, then prompts once for y / yes."""

    def __init__(
        self,
        *,
        read_line: Callable[[str], str] = input,
        write: Callable[[str], None] = _write_stdout,
        is_interactive: Callable[[], bool] = _stdin_is_interactive,
    ) -> None:
        self._read_line = read_line
        self._write = write
        self._is_interactive = is_interactive

    def confirm(self, summary: str) -> bool:
        """True only when the answer, stripped and lower-cased, is ``y`` or ``yes``.

        Without a terminal the summary is shown and the plan rejected without a prompt.
        EOF and Ctrl-C at the prompt reject; any other exception propagates.
        """
        interactive = self._is_interactive()
        self._write(summary)
        if not interactive:
            self._write(NO_TERMINAL)
            return False
        try:
            answer = self._read_line(PROMPT)
        except (EOFError, KeyboardInterrupt):
            self._write(NO_ANSWER)
            return False
        if answer.strip().lower() in _YES:
            return True
        self._write(REJECTED)
        return False
