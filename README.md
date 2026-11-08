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

The example uses mechanistic seeds and downsampled ReaxFF-MD trajectories for
n-heptane, methylcyclohexane and toluene. [Example inputs and outputs](examples/README.md)

| Run Explore | `labeling.gaussian.mode` | `loop.mode` | Behavior |
| --- | --- | --- | --- |
| Automatic on Linux | `local` | `auto` | Run Gaussian in the current environment and advance through the acquisition rounds. |
| Automatic on a cluster | `slurm` | `auto` | Submit Gaussian jobs with `sbatch`, poll for results and advance through the acquisition rounds. |
| Prepare inputs only | `manual` | `step` | Write Gaussian inputs. Supply the logs and run `tsgen explore resume` to collect them and advance. |

The supplied example uses `manual` and `step`. See the configuration for
[local Linux](docs/linux.md#local-linux-with-gaussian) or
[Slurm](docs/linux.md#slurm-cluster) to run Gaussian automatically.

Reproducing the manuscript's numerical results requires the original study
trajectories, split manifests, DFT labels and fitting inputs. This repository
provides the workflow code and downsampled Quickstart inputs.

## Relative-energy model

For a sampled configuration `i` of species `g`, the GP target in kcal/mol is

```text
y_i = 627.5094740631 * (E_SP(i) - E_Opt(g))
```

Both electronic energies are calculated at M06-2X/6-311G(d,p) in Hartree.
Each sampled geometry supplies a single-point energy, and all configurations
of the same species share a fixed reference energy from geometry optimization.

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

See the [study protocol](docs/protocol.md) for research inputs, independent
evaluation and matched-budget comparisons.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. TSGen is distributed
under the [MIT License](LICENSE).

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

`src/tsgen/` contains the installable workflows. `docs/` describes data contracts and execution. `tests/` contains automated regression checks for file handling, numerical methods, job execution and workflow recovery.
