# Seismic Tokenizer Future Work

## Near-term improvements
1. Integrate full rendered seismic overlay canvas (beyond preview statistics).
2. Stream partial output tiles to UI during running search for progressive visualization.
3. Add user-selectable colormap families and perceptual defaults in UI.
4. Persist run summary files (config, runtime metrics, output paths) per session.

## Performance roadmap
1. Add optional mixed precision inference where safe.
2. Add backend-specific tuning profiles (CPU, CUDA, MPS).
3. Improve data locality with tiled prefetch and optimized chunk traversal.
4. Add benchmark corpus and automated perf regression gates.

## Search quality roadmap
1. Add multi-token query composition (mean/max/weighted token sets).
2. Add uncertainty/confidence maps alongside similarity output.
3. Explore latent normalization/calibration strategies.
4. Add optional ANN index for repeated query acceleration.

## Reliability roadmap
1. Strengthen crash-resume behavior for long-running searches.
2. Add richer cancel semantics (retain/discard partial outputs by policy).
3. Add structured error taxonomy and user-facing remediation hints.
4. Add fault injection tests for worker process interruptions.

## Productization roadmap
1. Package app launchers for common desktop targets.
2. Add lightweight project/session manager in UI.
3. Add reproducible run manifests and exportable reports.
4. Introduce role-based review checklist for final release readiness.
