# Linux and Slurm operation

## Environment

Install TSGen into a Python 3.10 or newer environment as shown in the README.
Load the Gaussian environment in the shell that starts TSGen. For a standard
Gaussian 16 installation, a Bash setup has this form. Set the installation
and scratch paths for your machine.

```bash
export g16root=/opt/gaussian
source "$g16root/g16/bsd/g16.profile"
export GAUSS_SCRDIR="$HOME/scratch/gaussian"
mkdir -p "$GAUSS_SCRDIR"
command -v g16
command -v formchk
```

Gaussian receives input on stdin and writes its log on stdout. TSGen retains
each `input.gjf`, `input.log`, process exit status and job metadata.
Gaussian environment and scratch setup are described by the
[Gaussian installation instructions](https://ducc.du.ac.in/wp3/wp-content/uploads/2024/04/G16_UNIX.pdf).

Use `examples/quickstart.yaml` for the bundled initial acquisition example.
For a production study, supply the molecular seed geometries, unwrapped
trajectories, reference connectivity and split groups described in
[protocol.md](protocol.md).
Use the same DFT method, basis, charge and multiplicity for each sampled
structure and its equilibrium reference. Set explicit charge and multiplicity
for states that require them. The automatic setting derives a starting
assignment from electron parity.

## Local Linux with Gaussian

In `examples/quickstart.yaml` or your Explore YAML, set the execution
fields as follows. Keep the input paths and model settings in the chosen YAML.

```yaml
labeling:
  backend: gaussian
  gaussian:
    mode: local
    g16_cmd: g16
    route: '# M062X/6-311G(d,p) SP'
    reference_mode: optimized
    nproc: 8
    mem: 16GB
loop:
  mode: auto
```

Run `tsgen explore run --config examples/quickstart.yaml` for the bundled inputs.
Local execution runs
Gaussian jobs sequentially, with `nproc` cores allocated to each calculation.
The process inherits the shell's Gaussian environment. New species generate
a reference optimization alongside their sampled-geometry single points.
The next GP fit waits for the required successful labels and optimized
references. Use `tsgen explore status --run-dir PATH_TO_RUN` to inspect a run
and `tsgen explore resume --run-dir PATH_TO_RUN` to continue it.

Set the following fields in your trainset YAML to run the equilibrium
optimizations and relaxed scans locally.

```yaml
gaussian:
  mode: local
  g16_cmd: g16
  formchk_cmd: formchk
execution:
  submit: true
  wait_for_completion: true
```

Run `tsgen trainset run --config trainset.yaml` with your trainset YAML. The builder
first collects equilibrium optimizations, generates scans from the optimized
geometries and exports the validated fitting targets. Use
`tsgen trainset resume --run-dir PATH_TO_RUN` to advance an existing run.

For sensitivity analysis on the same workstation, set
`execution.mode: local` in your sensitivity YAML, configure its ReaxFF
executable and fitting inputs, then run
`tsgen sensitivity run --config sensitivity.yaml`.
INDEEDopt uses its own Python and TensorFlow environment.

## Slurm cluster

For automatic Explore, replace the `labeling` and `loop` sections of
`examples/quickstart.yaml` or your Explore YAML with the following settings.
Set the partition, walltime and module for your cluster. Match `nproc` to
`cpus_per_task` and leave memory for Gaussian process overhead in the Slurm allocation.

```yaml
labeling:
  backend: gaussian
  gaussian:
    mode: slurm
    g16_cmd: g16
    route: '# M062X/6-311G(d,p) SP'
    reference_mode: optimized
    nproc: 8
    mem: 8GB
    slurm:
      partition: compute
      time: '02:00:00'
      cpus_per_task: 8
      mem: 16G
      extra_lines:
        - module load gaussian/g16
loop:
  mode: auto
  poll_seconds: 60
  max_wait_seconds: 86400
```

```bash
tsgen explore run --config examples/quickstart.yaml
```

Use the appropriate module name for the cluster, or put the Gaussian profile
initialization in `extra_lines`. TSGen submits `submit.slurm` with `sbatch`,
retains the returned job ID and collects the Gaussian logs. The generated
scripts use LF line endings. Scratch setup must also be available in the
compute-node environment.

With `loop.mode: auto`, TSGen waits for the batch's single points and required
reference optimizations, collects valid labels and advances in the same
invocation. `max_wait_seconds` limits how long the driver waits for a batch
including queue time. Resume the existing run if the driver stops or reaches
that limit. Run the driver where your site allows scheduler polling for the
required duration. Use `loop.mode: step` to submit a batch and return.

```bash
tsgen explore resume --run-dir PATH_TO_RUN
```

For the trainset workflow, set `gaussian.mode: slurm` and
`execution.submit: true` in your trainset YAML. Set the site
resources and environment under its `gaussian.slurm` section.

For sensitivity calculations, set `execution.mode: slurm` and configure
`execution.slurm` in your sensitivity YAML. Inspect the generated pending
sample list and array scripts in the run's `slurm/` directory. Diagnostic
commands can collect failed-evaluation evidence and adjust concurrency on an
existing prepared run.

## Manual job execution and recovery

`mode: manual` prepares Gaussian inputs for external submission. Return each
Gaussian output to its job folder as `input.log` and resume the workflow.
`loop.mode: step` advances one acquisition step per invocation.
The independent test command uses the same Gaussian execution mode and
stores its results under `evaluation/`.

Explore writes species reference optimizations under `references/jobs/`.
When Gaussian reports an internal-coordinate error in a reference `Opt`,
TSGen prepares one retry with `Opt=Cartesian` in that job's `cartesian/`
subdirectory. Local and Slurm modes execute the retry automatically. In manual
mode, run the new input and return its log to the retry directory.

An optimization must converge with the same molecular graph to supply the
species energy zero. When a reference remains unavailable, TSGen reports the
affected species and log path and collects the valid labels from other species.
The final status reports how many acquisition labels failed. Inspect `input.log`,
stderr and the exit status for the affected jobs. The optimized species reference
stays fixed as later labels arrive. Each run uses its saved `config.yaml` on resume.
Existing records made with a sample-based energy zero require matching
optimized species energies before they can be used as the current relative
energy targets.
