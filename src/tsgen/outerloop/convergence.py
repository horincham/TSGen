from tsgen.outerloop.config import StopConfig


def decide(history: list[dict], cfg: StopConfig, dft_used: int) -> str:
    if dft_used >= cfg.dft_budget:
        return 'BUDGET_EXHAUSTED'
    configured = all(x is not None for x in (cfg.new_species_max, cfg.novelty_median_max, cfg.delta_ff_max))
    tail = history[-cfg.patience:]
    if configured and len(tail) == cfg.patience and all(
        m.get('n_candidates', 0) > 0 and m.get('validation_passed') is True
        and m['new_species'] <= cfg.new_species_max
        and m['novelty_median'] <= cfg.novelty_median_max
        and m['delta_ff'] <= cfg.delta_ff_max for m in tail
    ):
        return 'CONVERGED'
    if len(history) >= cfg.max_cycles:
        return 'MAX_CYCLES'
    return 'CONTINUE'
