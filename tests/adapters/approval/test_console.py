"""Tests for the console approval gate (offline: read_line, write and the tty check faked)."""

from __future__ import annotations

import sys

import pytest

from hedge_fund.adapters.approval.console import ConsoleApprovalGate
from hedge_fund.ports.approval import ApprovalGate

SUMMARY = "Proposal for 2026-10-23   risk: APPROVED\n"
PROMPT = "Approve and submit these orders? [y/N] "


class FakeConsole:
    """Scripted terminal: one answer (or exception) for read_line; records all I/O."""

    def __init__(self, answer: str | BaseException = "", *, interactive: bool = True) -> None:
        self.answer = answer
        self.interactive = interactive
        self.prompts: list[str] = []
        self.written: list[str] = []

    def read_line(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer

    def write(self, text: str) -> None:
        self.written.append(text)

    def is_interactive(self) -> bool:
        return self.interactive

    def gate(self) -> ConsoleApprovalGate:
        return ConsoleApprovalGate(
            read_line=self.read_line, write=self.write, is_interactive=self.is_interactive
        )


def test_satisfies_the_port() -> None:
    console = FakeConsole("y")
    gate: ApprovalGate = console.gate()
    assert gate.confirm(SUMMARY) is True


def test_no_terminal_rejects_without_a_prompt() -> None:
    console = FakeConsole("y", interactive=False)
    assert console.gate().confirm(SUMMARY) is False
    assert console.written == [SUMMARY, "No interactive terminal: rejected.\n"]
    assert console.prompts == []


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " yes ", "Yes\n"])
def test_yes_approves(answer: str) -> None:
    console = FakeConsole(answer)
    assert console.gate().confirm(SUMMARY) is True
    assert console.written == [SUMMARY]
    assert console.prompts == [PROMPT]


@pytest.mark.parametrize("answer", ["", "   ", "n", "no", "N", "nope", "ye", "yy", "yes please"])
def test_anything_else_rejects_without_a_re_prompt(answer: str) -> None:
    console = FakeConsole(answer)
    assert console.gate().confirm(SUMMARY) is False
    assert console.written == [SUMMARY, "Rejected.\n"]
    assert console.prompts == [PROMPT]


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()], ids=["eof", "ctrl-c"])
def test_no_answer_rejects(error: BaseException) -> None:
    console = FakeConsole(error)
    assert console.gate().confirm(SUMMARY) is False
    assert console.written == [SUMMARY, "\nNo answer: rejected.\n"]
    assert console.prompts == [PROMPT]


def test_other_exceptions_propagate() -> None:
    console = FakeConsole(RuntimeError("tty broke"))
    with pytest.raises(RuntimeError, match="tty broke"):
        console.gate().confirm(SUMMARY)
    assert console.written == [SUMMARY]
    assert console.prompts == [PROMPT]


@pytest.mark.parametrize("answer", ["y", "n"])
def test_summary_is_shown_before_the_prompt(answer: str) -> None:
    events: list[str] = []

    def read_line(prompt: str) -> str:
        events.append(f"prompt:{prompt}")
        return answer

    gate = ConsoleApprovalGate(
        read_line=read_line,
        write=lambda text: events.append(f"write:{text}"),
        is_interactive=lambda: True,
    )
    gate.confirm(SUMMARY)
    assert events[:2] == [f"write:{SUMMARY}", f"prompt:{PROMPT}"]


# --- defaults ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, *, tty: bool = False) -> None:
        self.tty = tty
        self.events: list[str] = []

    def write(self, text: str) -> int:
        self.events.append(text)
        return len(text)

    def flush(self) -> None:
        self.events.append("<flush>")

    def isatty(self) -> bool:
        return self.tty


def test_default_read_line_is_input() -> None:
    assert ConsoleApprovalGate()._read_line is input


def test_default_write_flushes_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = FakeStream()
    monkeypatch.setattr(sys, "stdout", stdout)
    ConsoleApprovalGate()._write("hello\n")
    assert stdout.events == ["hello\n", "<flush>"]


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [(FakeStream(tty=True), True), (FakeStream(tty=False), False), (None, False)],
    ids=["tty", "pipe", "missing"],
)
def test_default_terminal_check_reads_stdin_at_call_time(
    monkeypatch: pytest.MonkeyPatch, stdin: FakeStream | None, expected: bool
) -> None:
    gate = ConsoleApprovalGate()
    monkeypatch.setattr(sys, "stdin", stdin)
    assert gate._is_interactive() is expected


def test_package_exports() -> None:
    import hedge_fund.adapters.approval as approval

    assert approval.__all__ == ["ConsoleApprovalGate"]
    assert approval.ConsoleApprovalGate is ConsoleApprovalGate
