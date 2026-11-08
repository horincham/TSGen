# Study protocol and input contracts

## Candidate space and acquisition

1. Supply the mechanistic seeds and unwrapped all-atom XYZ trajectories for n-heptane, methylcyclohexane and toluene. Set the trajectory paths, stride and frame limits in your `explore.yaml` from the sampling record. Run from the repository root. Geometry reference and split-group paths resolve relative to the YAML file.
2. Supply the atom-mapped reference-bond JSON for the bond-extension test. `tsgen explore reference-bonds` creates a starting JSON from atom-mapped parents. Inspect the reference assignment and atom ordering for each candidate. The screening table records when the extension ratio could not be evaluated.
3. Run `tsgen explore prepare --config explore.yaml` to extract and screen the pool without preparing Gaussian jobs. The pool index records source file, source frame and species. Use the retained IDs to prepare `candidate_groups.csv` with columns `id,split_group` for all geometrically eligible structures. Assign adjacent trajectory frames and structurally similar configurations to the same group, including transitive links. The deterministic splitter keeps groups intact and stratifies by species and SOAP cluster. If exact split groups are unavailable, set `holdout.similarity_threshold` and `holdout.adjacent_frames` to define a new grouping protocol and remove `groups_file`. A split that cannot reach exactly 100 structures without dividing groups is rejected.
4. Run `tsgen explore run --config explore.yaml` and resume after each batch's logs are available. `holdout.csv`, `holdout.xyz` and `pool_split.csv` preserve the split. Gaussian 16 evaluates the selected structures at M06-2X/6-311G(d,p). It optimizes a reference geometry for each species at the same level. Successful optimized references are held fixed across batches. Reference optimizations and independent test jobs have separate accounting from the 400 acquisition single points. Failed single points can reduce the successful label count.
5. Run `tsgen explore label-holdout --run-dir <run_dir>` after acquisition. With manual execution, return each `input.log` to its corresponding job directory and invoke the same command to collect. Successful test labels are exported under `evaluation/labels/` and remain outside the acquisition labels. Run `tsgen explore evaluate --run-dir <run_dir>` for the final GP's train/test MAE and R², per-structure errors and novelty, and MAE by carbon count.

The study used species-resolved coverage selection, then geometric screening at 0.60 Å minimum separation, 1.50 maximum reference-bond extension ratio, and 45° minimum bonded angle. Nonperiodic SOAP uses a 5.0 Å cutoff, 0.5 Å width, eight radial functions, angular order six, and mean/standard-deviation pooling. The isotropic Matérn-3/2 GP uses L2-normalized SOAP and `alpha: 1.0e-6`. PCA is available as a separate projection. The acquisition budget is 10 rounds of up to 40 structures, with seed priority at initialization. The fitting export is capped at 120 DFT-labeled representatives.

Coverage cutoffs, acquisition weights, shortlist factor, random seed and scheduler resources are explicit settings. Use the original run values and manifests to reproduce the reported pool, split and label counts.

The current species fingerprint uses iterative graph-neighborhood refinement
(`species.fingerprint_version: wl_v2`). Set the same version in
`md.fingerprint_version` for the outer loop. This separates common constitutional
isomers that the earlier single-neighborhood fingerprint could merge. Molecular
graphs are inferred from the configured distance cutoff. Conformations of the
same graph share a reference. The grouping is independent of atom order.
Existing run snapshots without a version flag retain `legacy` identity on
resume. Use a new run for a changed identity scheme and supply matching optimized
references for imported label archives.

Successful reference optimizations write `optimized_reference.xyz` beside the
job log and an `energy_references.provenance.json` record in the run. This records
geometry, electronic state, method, energy and log hash. The reference is the
geometry obtained from that species' first successful Gaussian `Opt` job.
Vibrational frequency analysis is a separate calculation. The target is an
electronic energy difference from this fixed geometry. Reference provenance is
propagated into numerical evaluation and outer-cycle label handoffs.

## ReaxFF fitting targets and sensitivity

- `tsgen trainset plan --config trainset.yaml` stages equilibrium optimizations from the representative XYZ files specified in your YAML. `tsgen trainset resume --run-dir <run_dir>` advances to relaxed bond, angle and torsion scans and collects their logs. The paired fitting files are `exports/reaxff_trainset/geo` and `exports/reaxff_trainset/trainset.in`. Configure the scan grids and charge extraction in the same YAML.
- `tsgen sensitivity doctor --config sensitivity.yaml` checks the force field, fitting inputs and candidate parameter space specified in your YAML. Supply a standalone ReaxFF evaluator that produces `fort.13` and complete `fort.99` results. Sensitivity input paths resolve relative to the YAML. Use the exact study parameter-space manifest and fitting files to reproduce the reported 84 candidates.
- `tsgen sensitivity run --config sensitivity.yaml` runs the baseline check and stages evaluations. Manual mode writes `slurm/run_pending_samples.sh`. Run the script in the configured external-engine environment and call `tsgen sensitivity resume --run-dir <run_dir>` to collect and analyze. Local and Slurm execution modes are also supported.

The study included the initial vector among 10,000 parameter vectors and analyzed 10 repeated samples of 2,000 complete evaluations, drawn without replacement within each repeat. TSGen averages the per-repeat normalized HSIC scores, applies the 95% cumulative rule to the mean ranking of the total objective and reports per-parameter inclusion frequency. Signed estimates remain in the output tables. Negative estimates are clipped at zero for normalization and selection. Component rankings are available for energy, geometry and charge when those targets are present and nonconstant.

`tsgen sensitivity export-indeedopt --run-dir <run_dir>` exports the fitting inputs and the selected bounds. Run the reduced-space optimization with the external INDEEDopt installation. The study used center-maximin LHS, a fully connected network with two ReLU hidden layers and a linear output, Adam with MAE, Python 3.8.20, TensorFlow 2.10.1 and ten workers.

## Reactive MD and independent validation

Use [outerloop.md](outerloop.md) to import MD trajectories and matched force-field evaluations. TSGen operates on exported XYZ trajectories and file-based engine results. The initial three fuel trajectories drive candidate construction. The separate n-propylbenzene simulations validate the fitted force field.

The manuscript's AMS validation used a periodic box of 100 n-propylbenzene molecules at 0.15 g/cm³. It used minimization, 10 ps of equilibration at 300 K, and 200 ps of NVT production at 3500 K with a Nosé–Hoover chain thermostat, 100 fs coupling time and 0.1 fs integration step. Three production trajectories used different initial velocity seeds. Additional temperatures for kinetic analysis were 2300, 2600, 2900 and 3200 K. Reproduction requires the original cell, AMS input, velocity seeds and force field.
