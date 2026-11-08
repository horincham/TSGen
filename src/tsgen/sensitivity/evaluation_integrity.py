"""Match evaluated ReaxFF objectives, rather than only their summed loss."""
from collections import Counter,defaultdict
import math
import re


def _coordinate_identity(item):
    label=item.item_label
    if item.section=='CHARGE':
        match=re.search(r'\batom\s*:?\s*(\d+)',label,re.I)
        return ('atom',int(match[1])) if match else None
    if item.section=='GEOMETRY':
        if label.lower().startswith('geo '):
            fields=label.split()[3:-1]
            try:return ('atoms',tuple(int(v) for v in fields))
            except ValueError:return None
        match=re.search(r'(?:bond distance|valence angle|torsion angle)\s*:\s*(.*)',label,re.I)
        return ('atoms',tuple(int(v) for v in re.findall(r'\d+',match[1]))) if match else None
    if item.section=='ENERGY':
        terms=re.findall(r'([+-])\s*([A-Za-z_][A-Za-z0-9_.]*)\s*/\s*([-+]?\d+(?:\.\d*)?)',label)
        return ('energy',tuple((sign,ident,float(divisor)) for sign,ident,divisor in terms)) if terms else None
    return None


def _same_coordinate(first,second):
    a,b=_coordinate_identity(first),_coordinate_identity(second)
    if a is None or b is None:return a is None and b is None
    return a==b or (a[0]==b[0]=='atoms' and a[1]==tuple(reversed(b[1])))


def _row(item):
    return dict(section=item.section,structure_id=item.structure_id,
        item_label=' '.join(item.item_label.lower().split()),reference_value=item.reference_value,weight=item.weight)


def make_objective_manifest(losses,declared):
    expected=Counter(item.section for item in declared)
    observed=Counter(item.section for item in losses)
    if observed!=expected:
        raise ValueError(f'Incomplete objective coverage: declared {dict(expected)}, evaluated {dict(observed)}; missing/extra rows')
    buckets=defaultdict(list)
    for item in declared:
        buckets[(item.section,'+'.join(item.structure_ids))].append(item)
    for item in losses:
        key=(item.section,item.structure_id)
        matches=buckets.get(key,[])
        index=next((i for i,source in enumerate(matches)
            if math.isclose(source.reference_value,item.reference_value,rel_tol=1e-8,abs_tol=5.1e-5)
            and math.isclose(source.weight,item.weight,rel_tol=1e-8,abs_tol=5.1e-5)
            and _same_coordinate(source,item)),None)
        if index is None:
            raise ValueError(f'Objective identity/reference/weight not declared in trainset: {key}')
        matches.pop(index)
    return dict(version=1,objective_rows=[_row(item) for item in losses],coverage_complete=True)


def verify_objective_manifest(losses,manifest):
    expected=manifest['objective_rows']
    if len(losses)!=len(expected):
        raise ValueError(f'Objective rows changed: baseline {len(expected)}, sample {len(losses)}')
    for index,(item,source) in enumerate(zip(losses,expected)):
        actual=_row(item)
        if any(actual[k]!=source[k] for k in ['section','structure_id','item_label']):
            raise ValueError(f'Objective identity differs from baseline at row {index}')
        for field in ['reference_value','weight']:
            if not math.isclose(actual[field],source[field],rel_tol=1e-8,abs_tol=5.1e-5):
                raise ValueError(f'Objective {field} differs from baseline at row {index}')
