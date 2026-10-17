Paired Gamma BFD ENERGY workflow
===============================

``deterministic-LiH-pp-paired-energy-restart`` is one CUDA complex FP64 test,
conditional on Python h5py/numpy and a Git source checkout.  It initializes its
own model: no external trained checkpoint or orbital file is required.  Each
invocation retains a unique output subdirectory; no old evidence is deleted.

The recipe derives the existing LiH BFD Ne2/spin1+1/effective-charge1+1 input,
preserves both PP byte contents, seed37/model and91427/MC, timestep1e-4 and
Adam learning_rate1e-10.  It uses the existing skew periodic cell at Gamma,
explicit long-range ``LR_handler=ewald``/``LR_dim_cutoff=30`` (not optimizerLR),
no AE electron-ion double count, no teacher and no interval inference.  Only
ENERGY Adam/OFF is covered.  W1/crowd1/block1/step1 is a workflow proof, not a
statistical, convergence, energy-improvement or general materials claim.

Case84 actual proof: sourcefb82136b4, app9b41c772..., pilot2.7693s/RSS654564KiB,
two committed updates, actualQ12 each, max|ImVnl|.17068/.17852, truthful PP export,
finite parameters and nonzero accepted delta1.248925e-7.  The protected pilot
was reused, not rerun: split1/resume2 each1.8675s passed exact ROOT/optimizer,
sampler/RNG, all parameter-export datasets and nonclock history.  Case83's
earlier pre-training XML-precheck refusal is preserved in Task43 qualification.

The permanent helpers preserve that scientific logic; formatting and an
explicit pilot-helper location support registration.  Preparation/help/syntax,
unique-path and CMake discovery checks are not a new scientific rerun.  One
native child has60s; restart180s, orchestrator190s, CTest200s.  The existing
2GiB aggregate/256MiB reserve and explicit512MiB derivative device/HOST limits
remain; no cap growth, forced phase/alpha/seed or source/PP mutation.  Direct
parameter-only import/VMC, nonGamma, CPU, other optimizers and line search are
outside this qualification.  Checkpoint restart is the proven import route.

Separate fresh training import (manual, not the registered default)
-----------------------------------------------------------------

The pilot helper's optional ``--import-model MODEL`` retains the qualified
input's current Hamiltonian/PP/geometry/MC controls, removes initialization/seed,
and loads standalone values into NEW version0 training.  It is NOT sampler
restart, original-PP provenance or ordinary/interval inference admission.
The standalone file validates system/spins/ordered ionic positions/effective
charges/cell/twist and value/layout integrity, not the original PP bytes/Ewald.

Component88 at sourcef7619051a passed5263/1 assertions: exact loaded values,
new version0, Gamma G/L and complete same-grid complex energy, source unchanged;
CPU/wrong-kind/nonGamma requests refused.  Native89 at that source/app33e3f50c
returned0 with two fresh commits (2.7192s/RSS656436KiB), actualQ12, finite PP
exports and1,611,777 changed values in each update.  The original helper then
failed only while reporting a shadowed path; the failure is preserved.  A
reviewed separate OFFLINE validator completed all tail/input/evidence checks
without a scientific rerun or overwrite.  The fixed optional helper itself was
not rerun.  See Task43 validation_A_paired_pp_model_f7619051a and its separate
native89 qualification.  The registered test above still covers the original
internally initialized checkpoint-restart workflow only.
