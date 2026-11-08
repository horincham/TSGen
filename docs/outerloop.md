# Adaptive dataset expansion

`tsgen outerloop run --config outerloop.yaml` uses your outer YAML to freeze the initial
force field, fitting inputs, selected parameter bounds, DFT archive, and
preprocessing settings. Paths in this outer configuration are relative to
`project_root`, which is itself relative to the YAML. New runs use the next
`outerloop_run_000X` directory under `out_dir`. Set `run_dir` to use a
specific path instead. To continue the first default run, use
`tsgen outerloop resume --run-dir output/outerloop_run_0001`.

Set `initial.labels` and `initial.holdout_ids` from the actual Explore run
directory, such as `output/explore_run_0001`.
If independent test labels have been collected, set `initial.test_labels`
to `evaluation/labels/labels.csv` under that same directory.
Supply `initial.labels` as the Explore CSV export, or as an archive with
embedded geometries and `reference_hartree` for each successful label. The
reference must come from an optimized geometry for that species at the
acquisition DFT level. To keep independent test observations separate, provide
`initial.test_labels` or a CSV with an `id` column in `initial.holdout_ids`.
Use the same reference level for the archive, Explore and trainset routes.

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

AMS can be used externally for MD and export of unwrapped XYZ. A matched
older-force-field trajectory may accompany an MD result for diagnostics.
Only the updated force field's trajectories contribute to the candidate
ensemble. Candidate preprocessing uses the same chemical, diversity and
geometry settings as the linked Explore YAML. Include atom-mapped reference bond
entries for the outer candidates in the Explore geometry screening file.

TSGen performs DFT acquisition and relaxed-scan training-set construction
internally using the linked Explore and trainset YAMLs. With
`gaussian.mode: manual`, run the generated inputs and return their
`input.log` files to the corresponding job folders. Local or Slurm mode
uses the executable and scheduler settings from the linked YAMLs.

`summary.csv` records `new_species` (ΔS), `novelty_median` (median outer
SOAP nearest-neighbor distance), and `delta_ff` (mean absolute change in
relative force-field energies) for each assessed cycle. The energy comparison
uses the same species reference geometry for both force fields. TSGen picks
a reference geometry from the initial archive for each represented species
and freezes the first selected reference geometry for each new species.
These geometries are used only to define the origin of the force-field
energy comparison. Optimized DFT reference energies define the surrogate's
separate relative-energy targets.

Set `stop.max_cycles: 5` to cap assessment at five cycles. Set
`stop.acquire_on_final_cycle: true` to allow one last acquisition and dataset
update without fitting a sixth force field. After an update, `final.json`
distinguishes the archive used to fit the force field (`ffield_fit_archive`)
from the latest archive (`archive`). Threshold-based convergence also requires independent
validation limits and a matching validation report. A cap or exhausted
budget gives its own status.

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
