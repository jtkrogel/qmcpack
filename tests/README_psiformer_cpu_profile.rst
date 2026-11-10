CPU score-only product qualification
===================================

The optional PSIFORMER_CPU_SCORE_ONLY_PRODUCTS=1 training control selects an
explicit real FP64 CPU PsiFormer score-product owner for ECP score-Adam and
energy score-KFAC. Unset/0 preserves the existing complete product owner during
qualification. Other strings are rejected. An explicit unsupported request is
rejected rather than silently changing the objective or backend.

The score owner supports SCORE_JVP and SCORE_VJP, with no local-energy parameter
response and no kinetic-response tape. Its independently keyed scratch can be
shared by retained sample operators without borrowing complete-owner storage.
Full-response requests continue using the existing complete owner. Numerical
storage diagnostics account for absent kinetic storage and reject unsupported
products through the existing streaming capability contract.

Physical local energy is still computed by the complete native Hamiltonian and
its accepted ECP grid. This control omits unnecessary derivative capabilities
and grid retention; it does not omit the ECP energy or change its quadrature.
This first prototype does not remove repeated spatial/energy evaluations.

The initial qualification test filter is [cpu_score_only] in
src/QMCWaveFunctions/tests/test_psiformer_streaming_derivative.cpp. It compares
weighted score products against the complete owner, verifies smaller workspace
and parameter scratch, separate/shared owner identities and rejection of energy
response. The bounded correctness fixture has four electrons. Application
qualification uses N22/D16/W192, Gamma, with baseline and opt-in receipts,
committed sample counts, numerical trajectories, work counters and whole-step
memory. Later required bounds are N22/D64 and N44/D16,D64 at W192; no broader
screen is implied. This profile remains experimental until application review.
