# TSGen

Training Set Generator (TSGen) constructs DFT reference datasets and coordinates adaptive ReaxFF parameterization. It extracts molecular configurations from reactive trajectories, combines them with seeds, screens the candidate space and selects DFT batches using SOAP novelty and Gaussian-process uncertainty. Its outer loop returns fitted force fields to reactive MD and assesses the resulting expansion of the reference domain.

## Installation

Use Python 3.10 or newer in a dedicated environment. Run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
tsgen --help
```

## Quickstart

Run from the repository root after installation.

```bash
tsgen explore run --config examples/quickstart.yaml
```

This example uses the mechanistic seeds and every stored frame of the bundled n-heptane, methylcyclohexane and toluene ReaxFF-MD trajectories. It builds a pool of SOAP cluster representatives and initializes the surrogate with both seeds and trajectory configurations. When GP uncertainty has little spread, the next batch prioritizes supported representatives that expand coverage. [Example inputs and outputs](examples/README.md)

| Run Explore | `labeling.gaussian.mode` | `loop.mode` | Behavior |
| --- | --- | --- | --- |
| Automatic on Linux | `local` | `auto` | Run Gaussian in the current environment and advance through the acquisition rounds. |
| Automatic on a cluster | `slurm` | `auto` | Submit Gaussian jobs with `sbatch`, poll for results and advance through the acquisition rounds. |
| Prepare inputs only | `manual` | `step` | Write Gaussian inputs. Supply the logs and run `tsgen explore resume` to collect them and advance. |

The supplied example uses `manual` and `step`. For automatic Slurm execution, change both fields and set the site partition, resources and Gaussian module under `labeling.gaussian.slurm`. Slurm also supports `step` for submitting one batch per invocation. [Complete Slurm configuration](docs/linux.md#slurm-cluster) · [Local Linux configuration](docs/linux.md#local-linux-with-gaussian)

The Quickstart exercises acquisition and reference labeling with a small budget.
The study workflow uses an independent held-out set to measure prediction
accuracy. [Study protocol and input contracts](docs/protocol.md)

Reproducing the manuscript's numerical results requires the original study
trajectories, split manifests, DFT labels and fitting inputs. This repository
provides the workflow code and downsampled Quickstart inputs.

## Relative-energy model

For a sampled configuration `i` of species `g`, the GP target in kcal/mol is

```text
y_i = 627.5094740631 * (E_SP(i) - E_Opt(g))
```

Both electronic energies are calculated at M06-2X/6-311G(d,p) in Hartree. The sampled configuration supplies the single-point geometry. A converged geometry optimization supplies the common species reference. All configurations of that species share the fixed optimized reference energy. The reference record retains the optimized geometry, Gaussian route, charge, multiplicity and source log. Reference optimization jobs are separate from the acquisition single-point budget.

`labeling.gaussian.reference_mode: optimized` runs the species reference optimizations during acquisition. The optional `export.optimization.enabled` setting controls separate optimizations of selected fitting representatives.

The study used full L2-normalized SOAP descriptors, a Matérn-3/2 Gaussian process, 10 acquisition rounds of at most 40 single points, a grouped independent test set of 100 structures and at most 120 fitting representatives. The method and data contracts are described in [protocol.md](docs/protocol.md).

## Workflows

| Command group | Inputs and outputs |
| --- | --- |
| `explore` | Seed XYZ files and reactive trajectories → screened pool, fixed test manifest, DFT acquisition history, labels and fitting representatives. |
| `trainset` | Optimized structures and Gaussian relaxed scans → paired ReaxFF `geo` and `trainset.in`. |
| `sensitivity` | Force field, fitting targets and a ReaxFF evaluator → LHS responses, HSIC ranking and selected parameter bounds. |
| `outerloop` | Fitting datasets, successive force fields and MD results → assessed candidates, expanded reference archive and next fitting dataset. |
| `benchmark` | Saved screened pool, fixed test manifest and explicit DFT oracle → matched-budget TSGen, random and FPS metrics. |
| `diagnostics` | Production run artifacts → geometry checks, sensitivity diagnostics and Slurm concurrency adjustments. |

Gaussian 16 supplies the DFT calculations. ReaxFF evaluation, external INDEEDopt optimization and AMS MD require their respective installations and input files. See [outerloop.md](docs/outerloop.md) for the engine handoff contracts.

The Quickstart creates numbered run directories under `output/quickstart/`. Pass the existing run directory to `resume` to continue a workflow.

## Numerical evaluation

After acquisition, independently label the reserved test structures and evaluate the final surrogate.

```bash
tsgen explore label-holdout --run-dir PATH_TO_RUN
tsgen explore evaluate --run-dir PATH_TO_RUN
```

For matched-budget replay, use a screened study pool with a fixed held-out
split. Supply `oracle.csv` with `id,e_kcalmol_rel` and `initial_ids.txt` with
one initial training ID per line. Use a common optimized DFT reference for
each species in both sets.

```bash
tsgen benchmark run --run-dir PATH_TO_RUN --oracle oracle.csv \
  --initial-ids initial_ids.txt --budget 40 --budget 400 --seed 0 \
  --output benchmark.json
```

The JSON records train IDs, fixed test IDs, oracle domain size and MAE in kcal/mol. Replay budgets are independent of the prospective acquisition schedule.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. TSGen is distributed
under the [MIT License](LICENSE).

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

`src/tsgen/` contains the installable workflows. `docs/` describes data contracts and execution. `tests/` contains automated regression checks for file handling, numerical methods, job execution and workflow recovery.
