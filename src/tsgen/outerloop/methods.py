"""Conservative scientific identity for Gaussian routes used by the outer loop.

Only calculation tasks and population-output requests are ignored. In
particular, solvent, dispersion, SCF, grids, basis and other unrecognized
directives remain part of the identity. This deliberately rejects changes
whose scientific equivalence cannot be established from their spelling.
"""
from __future__ import annotations

import re


_TASK_OUTPUT = {'opt', 'sp', 'freq', 'nosymm', 'pop'}


def _method_identity(route: str) -> tuple[str, ...]:
    if not isinstance(route, str) or not route.strip():
        raise ValueError('Gaussian method route is empty')
    text = route.strip()
    if text.startswith('#'):
        text = text[1:].lstrip()
        # P/N/T select Gaussian printing detail, not the Hamiltonian.
        text = re.sub(r'^[pnt](?=\s|$)\s*', '', text, flags=re.I)
    text = re.sub(r'\s*=\s*', '=', text)
    tokens = []
    current = []
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                raise ValueError('Unbalanced parentheses in Gaussian method route')
        if char.isspace() and depth == 0:
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)
    if depth:
        raise ValueError('Unbalanced parentheses in Gaussian method route')
    if current:
        tokens.append(''.join(current))
    scientific = []
    for raw in tokens:
        token = re.sub(r'\s+', '', raw).lower()
        keyword = token.split('=', 1)[0].split('(', 1)[0]
        if keyword not in _TASK_OUTPUT:
            scientific.append(token)
    if not scientific:
        raise ValueError('Gaussian route has no scientific method')
    return tuple(sorted(scientific))


def validate_scan_methods(methods: list[str], cfg) -> None:
    """Reject a build protocol that changes the acquired labels' DFT method.

    ``cfg`` is the loaded TrainsetBuildCfg (with ``cfg.gaussian.reference_route``
    and ``cfg.gaussian.route``). Call before planning or submitting any Opt/scan
    jobs. Exceptions name the mismatched protocol without changing inputs.
    """
    identities = {_method_identity(method) for method in methods}
    if len(identities) != 1:
        raise ValueError('Historical labels must have one consistent Gaussian method')
    expected = next(iter(identities))
    for role, route in [('equilibrium', cfg.gaussian.reference_route),
                        ('scan', cfg.gaussian.route)]:
        if _method_identity(route) != expected:
            raise ValueError(f'{role} Gaussian method differs from acquired labels: {route}')
