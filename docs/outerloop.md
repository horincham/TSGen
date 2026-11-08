# Adaptive dataset expansion

Run `tsgen outerloop run --config outerloop.yaml` with the initial force field,
fitting inputs, selected parameter bounds and DFT archive. Paths resolve from
`project_root`, which is relative to the YAML. Runs are numbered under `out_dir`
unless you specify `run_dir`.

Set `initial.labels` to the Explore labels CSV or an archive containing
geometries and `reference_hartree` for each successful label. Use the same
DFT level for the archive, Explore and trainset routes. To keep independent
test observations separate, set `initial.test_labels` to the Explore run's
`evaluation/labels/labels.csv` or set `initial.holdout_ids` to a CSV with an
`id` column.

When transferring an Explore archive between machines for a new outer run,
point the input label table's `reference_provenance_path` to the relocated
provenance JSON and ensure its structure paths resolve from `project_root`.

## Manual engine tasks

The run pauses at engine tasks. Read the `pending_task` field from
`tsgen outerloop status --run-dir output/outerloop_run_0001`, and use that directory's
`request.json` to find the frozen inputs. Prepare a JSON manifest with an
`outputs` object that maps artifact names to file paths. Paths are relative
to the manifest. Import the result and resume.

```console
tsgen outerloop import-result --task-dir PATH_TO_TASK --manifest PATH_TO_RESULT_JSON
tsgen outerloop resume --run-dir output/outerloop_run_0001
```

| Stage | Required result artifacts |
| --- | --- |
| `optimize` | `ffield` and `fit_report` JSON with independently recomputed `objective` and `recomputed_objective`. Use INDEEDopt with the frozen training files and HSIC-selected parameter bounds. |
| `md` | `new_xyz` and optional `new_xyz_002`, `new_xyz_003`, etc. as unwrapped trajectory files generated with the updated force field. Supply `conditions` JSON with the `new` protocol, `coordinates: unwrapped`, and `input_hashes` keyed by MD input name. If no `md.inputs` were configured, supply `md_system` and `md_protocol` and include their SHA-256 digests as `system` and `protocol`. |
| `evaluate` | `energies` CSV with one row per requested candidate and columns `id,species_id,reference_id,old_energy,new_energy,old_reference,new_reference`. Evaluate both force fields on the same candidates and reference geometries in kcal/mol. Supply `validation` JSON with protocol and independent metrics, or an empty `metrics` object and a `reason` when these measurements are unavailable. |

Export unwrapped XYZ trajectories from the updated force field to build the
candidate ensemble. Preprocessing follows the linked Explore YAML. Include
atom-mapped reference bonds for outer candidates when bond-extension screening
is enabled.

TSGen performs DFT acquisition and relaxed-scan training-set construction
internally using the linked Explore and trainset YAMLs. With
`gaussian.mode: manual`, run the generated inputs and return their
`input.log` files to the corresponding job folders. Local or Slurm mode
uses the executable and scheduler settings from the linked YAMLs.

`summary.csv` records the new species count, median SOAP novelty and mean
absolute change in relative force-field energies for each cycle. Both force
fields use the same reference geometry for each species. The surrogate's
relative-energy targets use the optimized DFT references.

Set `stop.max_cycles` to cap assessment and `stop.acquire_on_final_cycle: true`
to allow a final dataset update without refitting. Threshold-based convergence
requires independent validation limits and a matching report. `final.json`
records the fitting archive (`ffield_fit_archive`) and the latest archive
(`archive`).

## Updating from an existing trajectory

To make one dataset update from an already generated MD trajectory, keep
`mode: manual` and add the following block. Starting the configuration
allocates a new numbered run directory.
For another update, use `next_initial` in `final.json` as the new initial
block and supply the next force field and its trajectories.

```yaml
existing_md:
  ffield: inputs/next_cycle/ffield
  trajectories:
    - inputs/next_cycle/trajectory.xyz
```

## Automatic engines

Set `mode: automatic` and configure `adapters.optimize`, `adapters.md` and
`adapters.evaluate` with external commands or native adapters. Use
`gaussian.mode: local` or `slurm` in both linked YAMLs.
`adapters.<stage>.command` is an argument list and can use the `{request}`
placeholder. Site-specific commands can use
`python -m tsgen.outerloop.worker --request {request} --site /path/to/site.yaml`
to stage inputs and import outputs. Each stage in the site YAML has
`commands` (a list of argument lists) and `outputs` (artifact names mapped
to paths under the task working directory).

The following adapter block enables native standalone ReaxFF energy
evaluation in manual or automatic mode.

```yaml
adapters:
  evaluate:
    inputs:
      reaxff: inputs/bin/reac
    native:
      kind: reaxff_evaluate
      reaxff_argv: ['{exe}']
      seed: 42
      atom_capacity: 1000
      timeout_seconds: 86400
```

The `indeedopt` native adapter also needs the `reaxff` input, a clean
`template_dir`, and `optimizer_argv` for the INDEEDopt installation.
INDEEDopt's Python and TensorFlow environment is configured separately
from TSGen. The `reaxff_md` native adapter takes `reaxff`, `md_geo`, and
`md_control` inputs, plus `coordinates: unwrapped` in its native settings.
Its system and protocol must match `md.inputs.system` and
`md.inputs.protocol` when these are set. This adapter uses standalone
ReaxFF MD. Use a site command or import results for AMS. Set the capacity
to match the compiled binary.
