"""Lossless cumulative ReaxFF geometry and canonical training-section merger."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import tempfile

from tsgen.outerloop.fitting import validate_fit_pair
from tsgen.trainset.io import read_frames_from_reaxff_geo
from tsgen.sensitivity.trainset_map import parse_geo_inventory


_SECTIONS = ('CHARGE', 'GEOMETRY', 'ENERGY')
_ID = r'[A-Za-z0-9_.]+'
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'
_TERM = re.compile(rf'([+-])\s*({_ID})\s*/\s*({_NUMBER})')


def _read(path: Path) -> str:
    # read_text's newline translation would alter the preserved old GEO blocks.
    return path.read_bytes().decode('utf-8')


def _geo_ids(path: Path) -> list[str]:
    """Reject incomplete blocks and atom rows the permissive reader skips."""
    active = False
    ident = None
    atom_ids: set[int] = set()
    expected: list[tuple[str, int]] = []
    for number, raw in enumerate(_read(path).splitlines(), 1):
        line = raw.strip()
        parts = line.split()
        if not parts or line.startswith(('#', '!')):
            continue
        key = parts[0]
        if key in {'BIOGRF', 'XTLGRF'}:
            if active:
                raise ValueError(f'{path}:{number}: missing GEO END')
            active, ident, atom_ids = True, None, set()
        elif not active:
            raise ValueError(f'{path}:{number}: expected BIOGRF/XTLGRF block')
        elif key == 'DESCRP':
            if ident is not None or len(parts) != 2 or not re.fullmatch(_ID, parts[1]):
                raise ValueError(f'{path}:{number}: invalid or duplicate DESCRP')
            ident = parts[1]
        elif key in {'ATOM', 'HETATM'}:
            try:
                index = int(parts[1])
                offset = 6 if not _is_number(parts[3]) and not _is_number(parts[4]) else (5 if not _is_number(parts[3]) else 3)
                coordinates = parts[offset:offset + 3]
                if index <= 0 or index in atom_ids or len(coordinates) != 3 or not all(_is_number(x) for x in coordinates):
                    raise ValueError('invalid atom')
            except (ValueError, IndexError) as exc:
                raise ValueError(f'{path}:{number}: malformed GEO atom') from exc
            atom_ids.add(index)
        elif key == 'END':
            if ident is None or not atom_ids or len(parts) != 1:
                raise ValueError(f'{path}:{number}: empty or unnamed GEO block')
            expected.append((ident, len(atom_ids)))
            active = False
    if active or not expected:
        raise ValueError(f'{path}: incomplete or empty GEO')
    inventory = parse_geo_inventory(path)  # also rejects duplicate identifiers
    frames = read_frames_from_reaxff_geo(path, preserve_ids=True)
    if [(i, len(f.symbols)) for i, f in frames] != expected or set(inventory) != {i for i, _ in expected}:
        raise ValueError(f'{path}: GEO parser/inventory mismatch')
    return [i for i, _ in expected]


def _is_number(token: str) -> bool:
    try:
        return math.isfinite(float(token.replace('D', 'E').replace('d', 'e')))
    except ValueError:
        return False


def _rewrite_row(raw: str, section: str, mapping: dict[str, str], allowed: set[str]) -> str:
    """Replace only identifier spans, never matching numbers or comments."""
    code = re.split(r'[#!]', raw, maxsplit=1)[0]
    tokens = list(re.finditer(r'\S+', code))
    spans = []
    if section in {'CHARGE', 'GEOMETRY'}:
        if len(tokens) not in ({4} if section == 'CHARGE' else {5, 6, 7}):
            raise ValueError(f'malformed {section} row: {code.strip()}')
        spans = [(tokens[0].start(), tokens[0].end(), tokens[0].group())]
        if not all(_is_number(t.group()) for t in tokens[1:]) or float(tokens[1].group().replace('D', 'E').replace('d', 'e')) == 0:
            raise ValueError(f'invalid {section} numeric value or zero weight')
        for token in tokens[2:-1]:
            if not token.group().isdigit() or int(token.group()) <= 0:
                raise ValueError(f'invalid {section} atom index')
    else:
        if len(tokens) < 3 or not _is_number(tokens[0].group()) or not _is_number(tokens[-1].group()):
            raise ValueError('malformed ENERGY row')
        if float(tokens[0].group().replace('D', 'E').replace('d', 'e')) == 0:
            raise ValueError('zero ENERGY weight')
        start, end = tokens[0].end(), tokens[-1].start()
        expression = code[start:end]
        cursor = 0
        for match in _TERM.finditer(expression):
            if expression[cursor:match.start()].strip() or not _is_number(match[3]) or float(match[3].replace('D', 'E').replace('d', 'e')) == 0:
                raise ValueError('malformed ENERGY expression or zero divisor')
            spans.append((start + match.start(2), start + match.end(2), match[2]))
            cursor = match.end()
        if not spans or expression[cursor:].strip():
            raise ValueError('malformed ENERGY expression')
    for start, end, identifier in reversed(spans):
        if identifier not in allowed:
            raise ValueError(f'trainset identifier absent from GEO: {identifier}')
        raw = raw[:start] + mapping.get(identifier, identifier) + raw[end:]
    return raw


def _train_sections(text: str, mapping: dict[str, str], allowed: set[str]):
    lines = text.splitlines(keepends=True)
    bodies: dict[str, list[str]] = {}
    ends: dict[int, str] = {}
    section = None
    for index, raw in enumerate(lines):
        code = re.split(r'[#!]', raw, maxsplit=1)[0].strip()
        if not code:
            if section:
                bodies[section].append(raw)
            continue
        upper = code.upper()
        if upper in _SECTIONS:
            if section or upper in bodies:
                raise ValueError('nested or repeated trainset section')
            section = upper
            bodies[section] = []
        elif section and upper == 'END' + section:
            ends[index] = section
            section = None
        elif section:
            rewritten = _rewrite_row(raw, section, mapping, allowed)
            bodies[section].append(rewritten)
        else:
            raise ValueError(f'unsupported trainset section or record: {code}')
    if section:
        raise ValueError(f'unclosed trainset section: {section}')
    return lines, bodies, ends


def merge_fitting_inputs(old_geo: Path, old_trainset: Path, new_geo: Path,
                         new_trainset: Path, output: Path) -> tuple[Path, Path, dict[str, str]]:
    """Write ``output/geo`` and ``output/trainset.in`` and return incoming ID map.

    All incoming IDs receive a deterministic namespace. Repeating the same call
    overwrites identical products; this is not a deduplication operation when
    already-merged inputs are subsequently supplied as a new prior dataset.
    Old GEO bytes and training rows remain intact. Only the three canonical
    target sections are accepted; unsupported records fail before publication.
    """
    old_geo, old_trainset, new_geo, new_trainset, output = map(Path, (old_geo, old_trainset, new_geo, new_trainset, output))
    destinations = (output / 'geo', output / 'trainset.in')
    if any(p.resolve() == target.resolve() for p in (old_geo, old_trainset, new_geo, new_trainset) for target in destinations):
        raise ValueError('Merged outputs must not overwrite source inputs')
    old_ids, new_ids = _geo_ids(old_geo), _geo_ids(new_geo)
    used = set(old_ids)
    mapping = {}
    for identifier in sorted(new_ids):
        salt = 0
        while True:
            digest = hashlib.sha256(f'{identifier}:{salt}'.encode()).hexdigest()[:12]
            candidate = 'ol_' + digest
            if candidate not in used:
                break
            salt += 1
        mapping[identifier] = candidate
        used.add(candidate)
    old_lines, old_bodies, old_ends = _train_sections(_read(old_trainset), {}, set(old_ids))
    _, additions, _ = _train_sections(_read(new_trainset), mapping, set(old_ids) | set(new_ids))
    merged_train = ''
    for index, raw in enumerate(old_lines):
        if index in old_ends:
            merged_train += ''.join(additions.get(old_ends[index], []))
        merged_train += raw
    if merged_train and not merged_train.endswith(('\n', '\r')):
        merged_train += '\n'
    for section in _SECTIONS:
        if section in additions and section not in old_bodies:
            merged_train += section + '\n' + ''.join(additions[section]) + 'END' + section + '\n'
    incoming = re.sub(r'(?m)^(\s*DESCRP\s+)([^\s]+)',
                      lambda m: m[1] + mapping[m[2]], _read(new_geo))
    original = old_geo.read_bytes()
    merged_geo = original + (b'' if original.endswith((b'\n', b'\r')) else b'\n') + incoming.encode('utf-8')
    # Validate staged products before replacing either existing result.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.merge-', dir=output) as stage:
        geo, train = Path(stage) / 'geo', Path(stage) / 'trainset.in'
        geo.write_bytes(merged_geo)
        train.write_bytes(merged_train.encode('utf-8'))
        validate_fit_pair(geo, train, [], (old_geo, old_trainset))
        geo.replace(destinations[0])
        train.replace(destinations[1])
    return *destinations, mapping
