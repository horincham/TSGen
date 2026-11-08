# Quickstart

## Inputs

The example combines mechanistic seed structures in `data/seeds/` with reactive MD
trajectories for n-heptane, methylcyclohexane and toluene in
`data/trajectories/`. The original trajectory files were too large to include,
so the bundled XYZ files contain downsampled trajectories. The example reads
all frames in the bundled files.

## Run the example

After installation, run from the repository root.

```bash
tsgen explore run --config examples/quickstart.yaml
```

The default manual mode prepares Gaussian inputs under
`output/quickstart/explore_run_0001/` on the first run. Run the single-point
jobs in `iter_*/gaussian/jobs/` and the species reference optimizations in
`references/jobs/`. Save each Gaussian output as `input.log` in its job
directory, then resume.

```bash
tsgen explore resume --run-dir output/quickstart/explore_run_0001
```

Repeat when the next batch is ready. For automatic execution, see the
[local Linux and Slurm instructions](../docs/linux.md).

## Inspect the outputs

- `exports/selected_for_label.xyz` contains the configurations selected for DFT.
- `exports/new_species_xyz/` contains successfully labeled non-seed configurations.
- `exports/optimized_references/` contains the optimized species reference structures.
- `labels/labels.csv` contains the collected energies and relative-energy labels.

For research runs and independent evaluation, see the
[study protocol](../docs/protocol.md).
