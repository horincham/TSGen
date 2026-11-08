"""Explicit atom-indexed reference connectivity, shared by all candidate gates."""
import json
import re
from pathlib import Path
import hashlib
from tsgen.data.xyz import read_xyz_multiframe
from tsgen.filters.geometry import infer_reference_bonds


def write_reference_bonds_from_xyz(xyz, output, key=None, bond_scale=1.25, max_bond_angstrom=2.25):
    """Build metadata from explicitly supplied reference parents for user review."""
    xyz, output = Path(xyz), Path(output)
    frames = read_xyz_multiframe(xyz)
    if not frames or (key is not None and len(frames) != 1):
        raise ValueError('Supply nonempty reference XYZ; --key requires exactly one frame')
    digest = hashlib.sha256(xyz.read_bytes()).hexdigest()
    data = {}
    for index, frame in enumerate(frames):
        reference_id = key or frame.get_id()
        if not reference_id or reference_id in data:
            raise ValueError('Reference XYZ needs a distinct id=... per frame, or --key for a single frame')
        bonds = infer_reference_bonds(frame.symbols, frame.coords, bond_scale, max_bond_angstrom)
        data[reference_id] = dict(symbols=frame.symbols, bonds=[list(pair) for pair in bonds],
            source_xyz=str(xyz.resolve()), source_xyz_sha256=digest, source_frame=index,
            graph_rule=dict(bond_scale=bond_scale, max_bond_angstrom=max_bond_angstrom,
                            radii_doi='10.1039/B801115J'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return data


def load_reference_bonds(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not data:
        raise ValueError('reference_bonds_file must contain a nonempty mapping')
    for key, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('symbols'), list) or not isinstance(entry.get('bonds'), list):
            raise ValueError(f'Invalid reference {key}: expected symbols and bonds lists')
        if not all(isinstance(s, str) for s in entry['symbols']):
            raise ValueError(f'Invalid reference symbols for {key}')
        seen = set()
        for pair in entry['bonds']:
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(type(i) is not int for i in pair) or pair[0] == pair[1] or
                    any(i < 0 or i >= len(entry['symbols']) for i in pair)):
                raise ValueError(f'Invalid reference bond {pair} for {key} (zero-based indices required)')
            canonical = tuple(sorted(pair))
            if canonical in seen:
                raise ValueError(f'Duplicate reference bond {pair} for {key}')
            seen.add(canonical)
    return data


def resolve_reference_bonds(frames, table, references):
    bonds, keys = [], []
    for frame, (_, row) in zip(frames, table.iterrows()):
        explicit = row.get('reference_id')
        if explicit is None or str(explicit) in ('', 'nan', 'None'):
            match = re.search(r'\b(?:reference_id|parent_id|parent)=([^\s]+)', frame.comment)
            explicit = match.group(1) if match else row.get('parent_id')
        if explicit is not None and str(explicit) not in ('', 'nan', 'None'):
            key = str(explicit)
        else:
            key = next((str(row.get(c)) for c in ('id', 'species_id') if str(row.get(c)) in references), None)
        if key not in references:
            raise ValueError(f'Missing reference bond mapping for candidate {row.get("id")}; set reference_id or parent_id')
        entry = references[key]
        if list(frame.symbols) != entry['symbols']:
            raise ValueError(f'Atom order/elements do not match reference {key}; supply an explicit atom-mapped graph')
        bonds.append(entry['bonds'])
        keys.append(key)
    return bonds, keys
