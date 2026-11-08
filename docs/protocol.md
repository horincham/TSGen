# Study protocol and input contracts

## Candidate space and acquisition

1. Supply the mechanistic seeds and unwrapped all-atom XYZ trajectories for n-heptane, methylcyclohexane and toluene. Set the trajectory paths, stride and frame limits in your `explore.yaml` from the sampling record. Run from the repository root. Geometry reference and split-group paths resolve relative to the YAML file.
2. Set `gating.reference_bonds_file` to an atom-mapped reference connectivity JSON for the bond-extension test. `tsgen explore reference-bonds --xyz parents.xyz --output reference_bonds.json` creates a starting JSON from atom-mapped parents. Inspect the reference assignment and atom ordering for each candidate. The reference connectivity specifies the pairs over which the ratio is evaluated. Each pair's sampled distance is divided by the sum of its two covalent radii. Inspect `rho_status` in the screening table for the evaluation status.
3. Run `tsgen explore prepare --config explore.yaml` to extract and screen the pool without preparing Gaussian jobs. Use `holdout.size: 0` for this step if you are generating the split-group file from scratch. The pool index records source file, source frame and species. Use the retained IDs to prepare `candidate_groups.csv` with columns `id,split_group` for all geometrically eligible structures. Assign adjacent trajectory frames and structurally similar configurations to the same group, including transitive links. For the study run, set `holdout.size: 100` and `holdout.groups_file` to that CSV. If exact split groups are unavailable, set `holdout.similarity_threshold` and `holdout.adjacent_frames` to define a new grouping protocol and omit `holdout.groups_file`. The deterministic splitter keeps groups intact, stratifies by species and SOAP cluster, and requires the configured test size to be reachable without dividing groups.
4. Run `tsgen explore run --config explore.yaml` and resume after each batch's logs are available. `holdout.csv`, `holdout.xyz` and `pool_split.csv` preserve the split. Gaussian 16 evaluates the selected structures at M06-2X/6-311G(d,p). It optimizes a reference geometry for each species at the same level. Successful optimized references are held fixed across batches. Reference optimizations and independent test jobs have separate accounting from the 400 acquisition single points. Failed single points can reduce the successful label count.
5. Run `tsgen explore label-holdout --run-dir <run_dir>` after acquisition. With manual execution, return each `input.log` to its corresponding job directory and invoke the same command to collect. Successful test labels are exported under `evaluation/labels/` and remain outside the acquisition labels. Run `tsgen explore evaluate --run-dir <run_dir>` for the final GP's train/test MAE and R², per-structure errors and novelty, and MAE by carbon count.

The study used species-resolved coverage selection, then geometric screening at 0.60 Å minimum separation, 1.50 maximum bond-extension ratio, and 45° minimum bonded angle. Nonperiodic SOAP uses a 5.0 Å cutoff, 0.5 Å width, eight radial functions, angular order six, and mean/standard-deviation pooling. The isotropic Matérn-3/2 GP uses full L2-normalized SOAP and `alpha: 1.0e-6`. For visualization, `descriptor.pca.enabled: true` writes a separate two-dimensional projection to `pool/features_pca_2d.npy`. The study acquisition budget is 10 rounds of up to 40 structures, with seed priority at initialization. The fitting export is capped at 120 DFT-labeled representatives.

Coverage cutoffs, acquisition weights, shortlist factor, random seed and scheduler resources are explicit settings. Use the original run values and manifests to reproduce the reported pool, split and label counts.

Use the same species fingerprint version in `species.fingerprint_version`
and `md.fingerprint_version` for Explore and the outer loop. Start a new run
if the species identity scheme changes and supply matching references for
imported labels. Successful Gaussian optimizations write `optimized_reference.xyz`
and record their provenance in `energy_references.provenance.json`.

## Matched-budget evaluation

Use a screened study run with a fixed test split and the study acquisition
policy. Supply `oracle.csv` with `id,e_kcalmol_rel` for each test ID and the
eligible acquisition structures in the replay domain. Use a common optimized
reference for each species across both sets. Put one starting training ID per
line in `initial_ids.txt`. Each budget includes those initial labels.

```bash
tsgen benchmark run --run-dir PATH_TO_RUN --oracle oracle.csv \
  --initial-ids initial_ids.txt --budget 40 --budget 400 --seed 0 \
  --output benchmark.json
```

## ReaxFF fitting targets and sensitivity

- `tsgen trainset plan --config trainset.yaml` stages equilibrium optimizations from the representative XYZ files specified in your YAML. `tsgen trainset resume --run-dir <run_dir>` advances to relaxed bond, angle and torsion scans and collects their logs. The paired fitting files are `exports/reaxff_trainset/geo` and `exports/reaxff_trainset/trainset.in`. Configure the scan grids and charge extraction in the same YAML.
- `tsgen sensitivity doctor --config sensitivity.yaml` checks the force field, fitting inputs and candidate parameter space specified in your YAML. Supply a standalone ReaxFF evaluator that produces `fort.13` and complete `fort.99` results. Sensitivity input paths resolve relative to the YAML. Use the exact study parameter-space manifest and fitting files to reproduce the reported 84 candidates.
- `tsgen sensitivity run --config sensitivity.yaml` runs the baseline check and stages evaluations. Manual mode writes `slurm/run_pending_samples.sh`. Run the script in the configured external-engine environment and call `tsgen sensitivity resume --run-dir <run_dir>` to collect and analyze. Local and Slurm execution modes are also supported.

The study used 10,000 parameter vectors and analyzed 10 repeated samples of 2,000 complete evaluations, drawn without replacement within each repeat. In TSGen, `sampling.center_initial_ffield` controls whether the initial force-field vector occupies one of the requested sample slots. TSGen averages the per-repeat normalized HSIC scores, applies the 95% cumulative rule to the mean ranking of the total objective and reports per-parameter inclusion frequency. Signed estimates remain in the output tables. Negative estimates are clipped at zero for normalization and selection. Component rankings are available for energy, geometry and charge when those targets are present and nonconstant.

`tsgen sensitivity export-indeedopt --run-dir <run_dir>` exports the fitting inputs and the selected bounds for optimization with INDEEDopt.

## Reactive MD and independent validation

Use the [outer-loop guide](outerloop.md) to import MD trajectories and
matched force-field evaluations. For independent n-propylbenzene validation,
follow the manuscript's simulation protocol using the original AMS inputs,
force field and initial velocity seeds.
