from __future__ import annotations

README_TEMPLATE = """# TSGenKit Review Bundle

This folder is produced by TSGen-Linux for human-in-the-loop review.

## What you have
- candidates.xyz: multi-frame XYZ pool (each frame comment line contains `id=...`)
- manifest.jsonl: per-structure metadata (id, iter, source, propose_params, hash, ...)
- features.csv: computed scores & visualization fields (id, score_u, score_r, score_d, S, PCA1, PCA2, filter_flags, ...)

## What you need to do
Create a file `decision.jsonl` in this folder.

Each line is a JSON object:
- id: candidate id
- action: "keep" or "drop"
- reason: short text (optional)
- tags: list of strings (optional)

Example:
{"id": "cand_00000001", "action": "keep", "reason": "high uE and novel region", "tags": ["beta_scission_like"]}
{"id": "cand_00000002", "action": "drop", "reason": "unphysical geometry", "tags": ["bad"]}

After writing `decision.jsonl`, return to TSGen-Linux and resume:
tsgen resume --run-dir <outputs/run_xxxx>

TSGen-Linux will read your decisions and continue the loop.
"""
