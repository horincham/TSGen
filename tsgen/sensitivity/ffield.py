from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

from tsgen.sensitivity.models import ParameterCoordinate, ParameterDefinition


SECTION_NAMES = {
    1: "general",
    2: "atom",
    3: "bond",
    4: "off_diagonal",
    5: "angle",
    6: "torsion",
    7: "hydrogen_bond",
}
VALUES_PER_BLOCK = {1: 1, 2: 32, 3: 16, 4: 6, 5: 7, 6: 7, 7: 4}
LINES_PER_BLOCK = {1: 1, 2: 4, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1}
KEY_FIELDS = {1: 0, 2: 0, 3: 2, 4: 2, 5: 3, 6: 4, 7: 3}
EXTRA_HEADER_LINES = {1: 0, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}
FLOAT_RE = re.compile(r"(?<!\S)[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")
INT_RE = re.compile(r"^\s*(\d+)")


@dataclass(frozen=True)
class _NumericField:
    coordinate: ParameterCoordinate
    line_index: int
    start: int
    end: int
    value: float


def _number(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _format_field_value(value: float, width: int) -> str:
    """Return the most accurate Fortran-readable representation fitting width."""
    candidates: set[str] = set()
    for digits in range(0, 13):
        fixed = f"{value:.{digits}f}"
        candidates.add(fixed)
        if fixed.startswith("0."):
            candidates.add(fixed[1:])
        elif fixed.startswith("-0."):
            candidates.add("-" + fixed[2:])
    for significant in range(1, 13):
        candidates.add(f"{value:.{significant}g}".upper())
    for digits in range(0, 10):
        scientific = f"{value:.{digits}E}"
        scientific = re.sub(r"E([+-])0+(\d+)$", r"E\1\2", scientific)
        candidates.add(scientific)

    fitted: list[tuple[float, int, int, str]] = []
    for candidate in candidates:
        if len(candidate) > width:
            continue
        try:
            parsed = _number(candidate)
        except ValueError:
            continue
        fitted.append(
            (
                abs(parsed - value),
                1 if "E" in candidate else 0,
                -len(candidate),
                candidate,
            )
        )
    if not fitted:
        raise ValueError(f"Value {value} cannot be represented in field width {width}")
    return min(fitted)[-1]


class FFieldDocument:
    def __init__(
        self,
        text: str,
        lines: list[str],
        parameters: list[ParameterDefinition],
        fields: dict[ParameterCoordinate, _NumericField],
        atom_symbols: dict[int, str],
    ):
        self._text = text
        self._lines = lines
        self.parameters = tuple(parameters)
        self._fields = fields
        self.atom_symbols = atom_symbols

    @classmethod
    def from_path(cls, path: str | Path) -> "FFieldDocument":
        with Path(path).open("r", encoding="utf-8", errors="strict", newline="") as handle:
            return cls.from_text(handle.read())

    @classmethod
    def from_text(cls, text: str) -> "FFieldDocument":
        lines = text.splitlines(keepends=True)
        if len(lines) < 2:
            raise ValueError("ffield must contain a title and seven parameter sections")
        cursor = 1
        parameters: list[ParameterDefinition] = []
        fields: dict[ParameterCoordinate, _NumericField] = {}
        atom_symbols: dict[int, str] = {}
        for section in range(1, 8):
            if cursor >= len(lines):
                raise ValueError(f"ffield ended before section {section}")
            count_match = INT_RE.match(lines[cursor])
            if not count_match:
                raise ValueError(f"ffield section {section} count is missing at line {cursor + 1}")
            block_count = int(count_match.group(1))
            cursor += 1 + EXTRA_HEADER_LINES[section]
            for block in range(1, block_count + 1):
                block_lines = list(range(cursor, cursor + LINES_PER_BLOCK[section]))
                if block_lines[-1] >= len(lines):
                    raise ValueError(f"ffield ended inside section {section}, block {block}")
                first_parts = lines[block_lines[0]].split()
                if section == 1:
                    block_key: tuple[str, ...] = ()
                elif section == 2:
                    block_key = (first_parts[0],)
                    atom_symbols[block] = first_parts[0]
                else:
                    block_key = tuple("*" if part == "0" else part for part in first_parts[: KEY_FIELDS[section]])
                found: list[tuple[int, re.Match[str]]] = []
                for line_offset, line_index in enumerate(block_lines):
                    matches = list(FLOAT_RE.finditer(lines[line_index]))
                    if line_offset == 0 and KEY_FIELDS[section]:
                        matches = matches[KEY_FIELDS[section] :]
                    found.extend((line_index, match) for match in matches)
                expected = VALUES_PER_BLOCK[section]
                if len(found) != expected:
                    raise ValueError(
                        f"ffield section {section}, block {block}: expected {expected} values, observed {len(found)}"
                    )
                for item, (line_index, match) in enumerate(found, start=1):
                    coordinate = ParameterCoordinate(section, block, item)
                    value = _number(match.group(0))
                    field = _NumericField(coordinate, line_index, match.start(), match.end(), value)
                    fields[coordinate] = field
                    name = f"{SECTION_NAMES[section]}_{item:02d}" if section != 1 else f"general_{block:03d}"
                    parameters.append(
                        ParameterDefinition(
                            coordinate,
                            SECTION_NAMES[section],
                            block_key,
                            name,
                            value,
                            line_index,
                            match.start(),
                            match.end(),
                        )
                    )
                cursor += LINES_PER_BLOCK[section]
        return cls(text, lines, parameters, fields, atom_symbols)

    def get(self, coordinate: ParameterCoordinate) -> float:
        try:
            return self._fields[coordinate].value
        except KeyError as exc:
            raise KeyError(f"Parameter {coordinate.parameter_id} is not present in ffield") from exc

    def quantize(self, coordinate: ParameterCoordinate, value: float) -> float:
        try:
            field = self._fields[coordinate]
        except KeyError as exc:
            raise KeyError(f"Parameter {coordinate.parameter_id} is not present in ffield") from exc
        return _number(_format_field_value(float(value), field.end - field.start))

    def render(self, updates: Mapping[ParameterCoordinate, float]) -> str:
        unknown = set(updates) - set(self._fields)
        if unknown:
            labels = ", ".join(sorted(item.parameter_id for item in unknown))
            raise ValueError(f"Cannot update coordinates absent from ffield: {labels}")
        lines = list(self._lines)
        grouped: dict[int, list[tuple[_NumericField, float]]] = defaultdict(list)
        for coordinate, value in updates.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite value for {coordinate.parameter_id}")
            grouped[self._fields[coordinate].line_index].append((self._fields[coordinate], float(value)))
        for line_index, replacements in grouped.items():
            line = lines[line_index]
            for field, value in sorted(replacements, key=lambda pair: pair[0].start, reverse=True):
                original = line[field.start : field.end]
                candidate = _format_field_value(value, len(original))
                line = line[: field.start] + candidate.rjust(len(original)) + line[field.end :]
            lines[line_index] = line
        return "".join(lines)

    def write(self, path: str | Path, updates: Mapping[ParameterCoordinate, float] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            handle.write(self.render(updates or {}))

    def catalog_rows(self) -> list[dict]:
        return [
            {
                "parameter_id": parameter.coordinate.parameter_id,
                "section": parameter.coordinate.section,
                "section_name": parameter.section_name,
                "block": parameter.coordinate.block,
                "item": parameter.coordinate.item,
                "block_key": "-".join(parameter.block_key),
                "parameter_name": parameter.parameter_name,
                "initial_value": parameter.initial_value,
            }
            for parameter in self.parameters
        ]
