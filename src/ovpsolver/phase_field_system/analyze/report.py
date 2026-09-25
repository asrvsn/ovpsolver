"""A table, printed the same way by every diagnostic.

Small on purpose: tables of at most a few hundred rows, printed at a terminal and
occasionally written as CSV, need no pandas. What the diagnostics share is the
verdict. A yes-or-no answer goes through :meth:`Report.check`, which prints the
same ``[ok]``/``[FAIL]`` line every time and decides the process exit status, so
an integration test can run a spec, run the diagnostics over its output, and let a
non-zero exit fail the build.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Printed to the left of a passing and a failing check.
OK, FAIL = "[ok]", "[FAIL]"


class Table:
    """Rows of one type, printed in aligned columns."""

    def __init__(self, *columns: str, formats: "dict[str, str] | None" = None) -> None:
        self.columns = columns
        self.formats = formats or {}
        self.rows: list[tuple[Any, ...]] = []

    def add(self, *values: Any) -> None:
        if len(values) != len(self.columns):
            raise ValueError(
                f"this table has {len(self.columns)} columns and was given "
                f"{len(values)} values"
            )
        self.rows.append(values)

    def __len__(self) -> int:
        return len(self.rows)

    def render(self) -> str:
        cells = [
            [self._format(column, value) for column, value in zip(self.columns, row)]
            for row in self.rows
        ]
        widths = [
            max([len(column), *(len(row[index]) for row in cells)])
            for index, column in enumerate(self.columns)
        ]

        def line(values: Iterable[str]) -> str:
            return "  ".join(
                value.rjust(width) for value, width in zip(values, widths)
            )

        return "\n".join(
            [
                line(self.columns),
                line("-" * width for width in widths),
                *(line(row) for row in cells),
            ]
        )

    def write_csv(self, path: "str | Path") -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        return destination

    def _format(self, column: str, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return format(value, self.formats.get(column, ".6g"))
        return str(value)


class Report:
    """What one diagnostic has to say, and whether it is bad news.

    Built up as the measurement goes and printed at the end, so that a check
    which fails halfway still shows the table that explains why.
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.lines: list[str] = []
        self.tables: list[Table] = []
        self.failures = 0

    ## Saying things

    def note(self, text: str = "") -> None:
        self.lines.append(text)

    def field(self, label: str, value: Any) -> None:
        self.lines.append(f"{label}: {value}")

    def table(self, table: Table) -> Table:
        self.tables.append(table)
        return table

    def check(self, passed: bool, description: str, detail: str = "") -> bool:
        """One yes-or-no verdict, which is what an integration test reads."""

        self.failures += not passed
        suffix = f"  ({detail})" if detail else ""
        self.lines.append(f"{OK if passed else FAIL} {description}{suffix}")
        return passed

    ## Getting them out

    @property
    def ok(self) -> bool:
        return self.failures == 0

    def render(self) -> str:
        blocks = ["\n".join(self.lines)] if self.lines else []
        blocks.extend(table.render() for table in self.tables if len(table))
        body = "\n\n".join(block for block in blocks if block)
        return f"{self.title}\n{'=' * len(self.title)}\n\n{body}"

    def emit(self, save: "str | Path | None" = None) -> int:
        """Print, optionally write the first table as CSV, and return an exit code."""

        print(self.render())
        if save is not None and self.tables:
            print(f"\nwrote: {self.tables[0].write_csv(save)}")
        if not self.ok:
            print(f"\n{self.failures} check(s) failed")
        return int(not self.ok)


__all__ = ["FAIL", "OK", "Report", "Table"]
