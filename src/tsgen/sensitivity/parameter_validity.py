"""Conservative support rules for the standalone ReaxFF parameter layout."""
from tsgen.sensitivity.models import ParameterDefinition,TrainingItem


def exclusion_reason(parameter: ParameterDefinition, document, inventory: dict,
                     items: list[TrainingItem], exclude_inactive: bool = True) -> str:
    used={ident for item in items for ident in item.structure_ids}
    geometries=[symbols for name,symbols in inventory.items() if not used or name in used]
    present=set().union(*geometries) if geometries else set()
    coordinate=parameter.coordinate
    if coordinate.section==2:
        symbols={document.atom_symbols.get(coordinate.block,'')}
    elif coordinate.section>=3:
        symbols={document.atom_symbols.get(int(key),'') for key in parameter.block_key if key.isdigit()}
    else:
        symbols=set()
    if symbols and (not symbols<=present or not any(symbols<=group for group in geometries)):
        return 'excluded_absent_element'
    if exclude_inactive:
        # The standalone reader only applies positive off-diagonal overrides;
        # nonpositive values retain combining rules, rather than being radii.
        if coordinate.section==4 and parameter.initial_value<=0:
            return 'excluded_disabled_override'
        # Explicit engine-profile rules: no continuous sensitivity for switches
        # or fields unused by the classical standalone implementation.
        if (coordinate.section==3 and coordinate.item in {6,15,16}) or (
                coordinate.section==6 and coordinate.item in {6,7}) or (
                coordinate.section==2 and coordinate.item in {13,16,23,24,27}):
            return 'excluded_discrete_or_unused'
    return ''
