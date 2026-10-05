# Working agreements

Read README.md and docs/scientific-contract.md before changing computation.
The published paper and hashed revision sources define the scientific contract.
Preserve paper/scripts/ as historical evidence; make portable adaptations in
paper/run_analysis.py and record them. Do not silently change stage coding,
processing rate, BSI scale, thresholds, episode definitions, or learned-score
cross-validation. Runtime kernels are extracted from the archived source;
changes require explicit numerical parity evaluation and documentation.

Keep raw recordings, participant tables, manifests with clinical identifiers,
predictions, caches, and large generated outputs outside Git. Only synthetic
examples are bundled. The author supplied the paper PDF for redistribution;
its copyright remains separate from the code license.

Use a project-local environment. Run pytest and build the package for runtime
changes. Full cohort analyses require authorized inputs and an external
workspace. Keep verification observations distinct from source-result checks.
Publication or remote mutations require user authorization for that scope.
