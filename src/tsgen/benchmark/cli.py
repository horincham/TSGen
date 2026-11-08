from pathlib import Path
import json

import numpy as np
import pandas as pd
import typer

from tsgen.benchmark.matched import run_benchmark
from tsgen.explore.config import load_explore_config


app = typer.Typer(add_completion=False, help="Evaluate matched label budgets on a fixed holdout.")


@app.callback()
def benchmark() -> None:
    pass


@app.command()
def run(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
    oracle: Path = typer.Option(..., exists=True, dir_okay=False, help="CSV with id and e_kcalmol_rel."),
    initial_ids: Path = typer.Option(..., exists=True, dir_okay=False, help="One initial structure ID per line."),
    budget: list[int] = typer.Option(..., help="Total training labels at each checkpoint. Repeat in ascending order."),
    output: Path = typer.Option(..., dir_okay=False),
    seed: int = typer.Option(0, min=0),
) -> None:
    """Compare TSGen, random selection and FPS using precomputed DFT energies."""
    try:
        cfg = load_explore_config(run_dir / "config.yaml")
        if cfg.descriptor.backend != "soap" or cfg.descriptor.normalization != "l2":
            raise ValueError("Run must use L2-normalized SOAP descriptors")
        pool = pd.read_csv(run_dir / "pool/pool_index.csv", dtype={"id": str})
        features = np.load(run_dir / "pool/features.npy", allow_pickle=False)
        gating = pd.read_csv(run_dir / "pool/geometry_screening.csv", dtype={"id": str}).set_index("id")
        passed = gating["passed_gating"].reindex(pool["id"]).eq(True).to_numpy()
        if len(pool) != len(features) or not passed.any():
            raise ValueError("Run must contain aligned pool descriptors and screened candidates")
        split = pd.read_csv(run_dir / "pool_split.csv", dtype={"id": str, "split_group": str})
        holdout = pd.read_csv(run_dir / "holdout.csv", dtype={"id": str})
        if len(holdout) != cfg.holdout.size or set(holdout["id"]) != set(split.loc[split["is_holdout"], "id"]):
            raise ValueError("Holdout manifest does not match the configured grouped split")
        if split["id"].duplicated().any() or set(split["id"]) != set(pool.loc[passed, "id"]):
            raise ValueError("Grouped split does not cover the screened pool exactly")
        if split.groupby("split_group")["is_holdout"].nunique().gt(1).any():
            raise ValueError("A candidate group spans both acquisition and test")
        energies = pd.read_csv(oracle, dtype={"id": str}).set_index("id")["e_kcalmol_rel"]
        ids = [line.strip() for line in initial_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = run_benchmark(
            pool.loc[passed], features[passed], energies, holdout["id"].tolist(), ids, budget,
            seed=seed, acquisition_cfg=cfg.acquisition, surrogate_cfg=cfg.surrogate,
        )
    except (ValueError, OSError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = {
        "test_ids": holdout["id"].tolist(),
        "screened_pool_size": int(passed.sum()),
        "oracle_acquisition_size": len(energies) - len(holdout),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except FileExistsError as exc:
        raise typer.BadParameter(f"Output file already exists: {output}") from exc
    typer.echo(f"benchmark={output}")
