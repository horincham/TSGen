# Initial acquisition Quickstart

The bundled inputs follow the initial acquisition setup in Methods §2.2.

| Input | Location | Contents |
| --- | --- | --- |
| Mechanistic seeds | `data/seeds/` | XYZ structures spanning small molecules, alkanes, cycloalkanes and aromatics |
| n-Heptane MD | `data/trajectories/n_heptane.xyz` | XYZ trajectory |
| Methylcyclohexane MD | `data/trajectories/methylcyclohexane.xyz` | XYZ trajectory |
| Toluene MD | `data/trajectories/toluene.xyz` | XYZ trajectory |

The bundled XYZs are downsampled exports of the original trajectories. The example reads all stored frames. Set a trajectory source's `stride` above one to sample at regular intervals within that file, or set `max_frames` to cap the number retained per file after striding.

Run from the repository root after installing TSGen.

```bash
tsgen explore run --config examples/quickstart.yaml
```

The example splits trajectory frames into connected components and retains H/C
molecules with at most eight carbons. Before computing SOAP, it excludes
odd-electron fragments and requires a conventional neutral valence assignment
with compatible bond lengths and local geometry. This screen uses H valence
one, C valence four, and C–C bond orders one through three. The overlapping
C–C windows are 1.27–1.72, 1.20–1.48 and 1.10–1.30 Å. The C–H window is
0.90–1.25 Å, and a carbon with two neighbors must have a bonded angle of at
least 150°. The pool writes rejection reasons to `pool/pool_prefilter_dropped.csv`.

The example clusters deduplicated trajectory configurations within each
perceived species in SOAP space. The nearest actual trajectory geometry to
each cluster center represents that cluster. `support_frames` counts distinct
source-file/frame pairs among its deduplicated members. It describes sampling
support in the input trajectories.

The example sets two rounds of up to six single-point calculations each and
uses a holdout size of zero.

Manual mode writes the first batch without requiring a Gaussian installation. The console reports input loading, descriptor calculation, screening, job preparation and the wait for DFT logs. New Quickstart runs are numbered under `output/quickstart/`, starting at `explore_run_0001`.

The first batch assigns up to three slots to seeds and uses the remaining
slots for trajectory representatives. Where available, the batch pairs seeds
with trajectory geometries of the same species. After labeling, the next
batch uses GP scores when the candidate uncertainties have sufficient spread.
With flat uncertainties it uses trajectory support and descriptor separation
from labeled and already selected structures. Regular slots require at least
two distinct supporting frames. One slot is reserved for a previously
unrepresented species when available, with priority for supported species.

The terminal reports `Representative bootstrap`, then
`representative_gp` or `representative_coverage`. The latter two are also
recorded in the second round's `scored.csv`. Each selected trajectory
geometry supplies the SP input. When a species needs a reference Opt, the
calculation can start from the representative with the highest trajectory
support and the same species identity and electronic state.

- `pool/pool_index.csv` records input provenance, species, support frames and screening decisions.
- `exports/filtered_pool.xyz` contains the geometries that passed screening.
- `iter_000/selected.xyz` contains the first batch and `exports/selected_for_label.xyz` contains the cumulative selections.
- `iter_000/gaussian/jobs/` contains the sampled-geometry single-point inputs.
- `references/jobs/` contains the species reference optimization inputs.
- `exports/new_species_xyz/` contains successfully labeled non-seed configurations at their sampled geometries.
- `exports/optimized_references/` contains one converged optimized XYZ per species and an index with energies, electronic states and Gaussian provenance.
- `run.log.jsonl` retains the progress messages.

Run each Gaussian input in its job directory and save the result as `input.log`. Resume to collect energies and advance to the next round.

```bash
tsgen explore resume --run-dir output/quickstart/explore_run_0001
```

After collection, `exports/labeled_ok.xyz` contains the successfully labeled geometries. The second round saves its fitted acquisition model to `surrogate/model_iter_001.pkl`. Resume again after the second batch completes to collect its results and save `surrogate/model_final.pkl` using the available training labels.

Start a new run to use representative acquisition. To resume an existing
Quickstart directory, keep its saved sampling and selection settings. If its
filenames predate the current distribution, update the trajectory paths to the
matching renamed files.

Local execution and Slurm submission use the same configuration and data paths. Change `labeling.gaussian.mode` and the resources for your machine. Use `loop.mode: auto` to continue through both rounds, or `step` to advance a batch at a time. Start a new run with `explore run` after editing the template. Existing runs use their own saved configuration. [Execution settings](../docs/linux.md)

## Inspect the generated structures

Open `exports/new_species_xyz/` to inspect the sampled geometries from the
trajectories. Find the optimized geometry for each species in
`exports/optimized_references/`. The species ID links the sampled geometry,
the reference and the relative-energy label. Use `labels/labels.csv` to
compare sampled configurations against their shared reference.

To regenerate the optimized-reference export for a saved run, use:

```bash
tsgen explore export-references --run-dir output/quickstart/explore_run_0001
```

This command reads the saved reference energies and provenance without
rebuilding the candidate pool or submitting Gaussian jobs.

## Study input layout

For a study run, point your YAML paths to the production data. One possible layout is:

```text
inputs/
  seeds/                 Molecular seed XYZ files
  trajectories/          Unwrapped reactive-MD XYZ trajectories
  reference_bonds.json   Atom-mapped reference bonds for geometric screening
  candidate_groups.csv  Candidate IDs and their shared split groups
  fitting/               Initial force field, geo, trainset.in, control, params-ml
data/
  representatives/       Representative optimized XYZ files for relaxed scans
  reaxff/                Sensitivity inputs and the standalone ReaxFF executable
```

Run `tsgen explore prepare --config explore.yaml` with your study YAML to write the candidate IDs before creating a split-group table. Each row in `candidate_groups.csv` supplies `id,split_group`. Assign related trajectory frames and similar structures to a common group. Use `tsgen explore reference-bonds --help` for the reference-connectivity format.

The fitting inputs depend on the external ReaxFF implementation. Preserve the geometry identifiers and target weights when transferring them between workflows.
