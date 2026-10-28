CPU Jastrow product workflows for diamond8e
===========================================

These complete inputs use the packaged two-carbon primitive diamond cell,
eight BFD valence electrons, Gamma orbitals and periodic Coulomb with
LR_dim_cutoff=30. Run from this directory so the relative orbital/BFD paths
resolve. Use a real, full-precision CPU build.

The student is exp(J1+J2+J_N) times the existing determinant expansion.
``psiformer_periodic_jastrow_v1`` selects ``zero_residual_pair_v1``: two
log-length parameters start at exactly zero, giving exactly zero J_N and
spatial derivatives. Their energy-stage scores are nonzero. Fixed reference
lengths default to 1 bohr; the 1.5 bohr cutoff lies inside this skew cell's
conservative reciprocal-face sphere (about 1.9475 bohr). The compact correction
adds zero electron-electron cusp slope for every learned positive length.
The analytic neural cusp is replaced; ordinary J1/J2 and orbital envelopes remain.

Inputs and execution
--------------------

``orbital_pretrain.xml`` performs orbital-MSE Adam against the frozen SJ
reference orbitals while sampling the full student product. It masks neural
pair lengths and all traditional coefficients off. Each energy input activates
the pair lengths and obeys the existing ``optimize`` attribute on each ordinary
``coefficients`` element. The ``j_frozen`` and ``j_active`` files demonstrate
both selections for Adam and KFAC. No determinant mixing coefficient or
normalization input is introduced.

The neural component uses ``optimize=no`` because that attribute selects its
legacy scalar-variable interface. WFTrain registers neural tensors directly;
the new profile rejects legacy scalar registration. Traditional Jastrow
``coefficients optimize=yes/no`` remain the energy-stage controls.

Use one native process for a stage handoff::

  python3 product_workflow.py handoff orbital_pretrain.xml \
    energy_adam_j_active.xml --output pretrain_to_adam.xml
  OMP_NUM_THREADS=1 /path/to/qmcpack pretrain_to_adam.xml

Substitute any of the four energy inputs. The energy input supplies the
physical wavefunction and its conventional coefficient selections. The launcher
copies the orbital-pretraining QMC block before the energy block. QMCPACK carries
the fitted wavefunction, walker configurations and RNG streams between drivers;
each objective creates fresh optimizer history. Both stages retain their stated
warmup/steps/substeps controls. Running two separate processes without a checkpoint
or model association instead starts a new sampler.

The checked-in examples use PBYP and line_search=off, with 16 walkers,
64 warmup sweeps and 10 updates per stage. Generate CPU real FP64 all-particle
student moves with ``--move alle``. This uses the complete neural/J1/J2 product
for VMC, orbital pretraining and ENERGY Adam/KFAC. The separate conventional
Slater-Jastrow reference input retains PBYP. The physical BFD Hamiltonian and
eight valence electrons are unchanged. To generate short all-particle controls::

  python3 make_examples.py --asset-dir .. --output-dir short_cases --acceptance --move alle
  python3 product_workflow.py handoff short_cases/orbital_pretrain.xml \
    short_cases/energy_kfac_j_active.xml --output short_cases/pipeline.xml

The acceptance preset uses 3 updates per stage, 4 walkers, 4 warmup steps and
2 sampled steps per update. It measures finite complete-product observations,
mask enforcement, publications and persistence; it establishes no convergence.
Omit ``--move alle`` to reproduce the PBYP controls. All-particle proposals
stage only conventional Jastrow components in private particle banks; acceptance
copies their native caches, while rejection preserves accepted values. J1/J2
coefficient publications are reflected in the next proposal. Accelerator and
complex/mixed-precision conventional Jastrow proposals remain outside this route.

Optimizer and memory conventions
--------------------------------

Energy inputs explicitly request raw physical score covariance. Local energies,
kinetic cross terms and BFD virtual-particle ratios come from the complete
student wavefunction. Full local-energy parameter response is outside this
profile. Pretraining remains orbital-MSE Adam.

KFAC uses the existing neural affine/fallback blocks and one bounded conventional
Jastrow affine block. Each complete configuration contributes its full aggregated
coefficient score vector with scalar sensitivity 1. ``native_score`` uses the
uncentered second moment E[O O^T]; ``registered_gaussian`` gives twice that
curvature. A small dense oracle tests both conventions. Cross curvature between
neural and conventional blocks is omitted. Active coefficients use KFAC itself;
there is no scalar-Adam companion. At most 256 conventional coefficients are
admitted, including frozen banks; the dense factor uses the selected subset.
This diamond input has 24 (a 4608-byte coefficient factor).

Neural tensors stay outside legacy VariableSet. Replay reuses a Jastrow-only
particle/component bank and fixed O(P) score contractions. One accepted iteration
prepares private states, agrees readiness across ranks, then publishes one joint
version while preserving the native resource/model identity. The driver refreshes
walker caches before the next sampled iteration.

Models, relocation and restart
------------------------------

Native iteration bundles contain theta, neural log lengths, selected conventional
coefficients, optimizer history and rank-local sampler state. Frozen conventional
values belong to the matching wavefunction association and immutable model
identity. ``R.training_info.csv``, ``R.training/``, ``R.inference/`` and
``R.last_checkpoint.h5`` follow the existing WFTrain output conventions.

For an exact same-stage restart, add ``restart_path`` to the training controls
and keep the physical input, masks, objective, optimizer and sampling topology
unchanged. Extend ``iterations`` if needed. The checkpoint can be its complete
``*.checkpoint`` directory or the latest-pointer HDF5 file. Resume accepts existing
inference containers and refuses future numbered iteration artifacts. Runtime
parameter generations are rebased by the existing checkpoint mechanism.

Each ``*.parameters.h5`` remains a directly importable native neural model. Its
``jastrow_neural_product`` group additionally stores canonical joint values and
``complete_wavefunction_xml`` with current J1/J2 coefficients. Loading only the
``psiformer`` block loads neural weights; sibling Jastrow components must come
from the matching association. Extract a complete inference input::

  python3 product_workflow.py extract R.training/R.i0000002.parameters.h5 \
    initial_product_vmc.xml --output current_product_vmc.xml --asset-dir ..
  OMP_NUM_THREADS=1 /path/to/qmcpack current_product_vmc.xml

The extractor binds the relocated model file explicitly and can relocate the
packaged orbital/BFD assets. This model import restores a physical wavefunction;
it starts fresh sampler/optimizer state. It does not replace an exact checkpoint
restart. The artifact records correlation policy, fixed references/cutoff and
conversion provenance when present; it preserves trained neural log lengths.

Legacy bare-neural models require explicit conversion::

  python3 convert_legacy_model.py legacy.parameters.h5 converted.parameters.h5 \
    --cutoff 1.5
  python3 product_workflow.py extract converted.parameters.h5 initial_product_vmc.xml \
    --output converted_product.xml --asset-dir .. --pin-template-jastrows

The converter
accepts a native standalone real Gamma periodic pseudopotential model artifact,
preserves compatible determinant tensors, replaces the old correlation policy
with exactly zero residual log lengths and records the immutable source SHA256.
The explicit association option pins the supplied traditional coefficients.
This is a new physical model with fresh optimizer/sampler history. Existing
residual-profile models are rejected by the legacy converter, preventing a trained
J_N from being silently reset.

Validation and interpretation
-----------------------------

Focused registered tests cover exact-zero/coalescence/cutoff behavior, finite-math
admission, seeded CPU value/VGL/score derivatives, full-product finite differences,
kinetic cross terms, actual accepted carbon BFD quadrature ratios, teacher-bank
isolation, masks, private rollback, model relocation and a dense Jastrow metric.
Native qualification covers short pretrain-to-Adam/KFAC workflows with J frozen
and active, scheduled inference, same-stage restart and a modest MPI/crowd case.

``sj_reference_vmc.xml`` supplies a matching short reference diagnostic.
Compare complete-product energy and variance after pretraining as well as orbital
MSE. A few orbital updates are not a claim of SJ-quality energy or convergence.
GPU/complex profiles, all-electron generalization, DMC and larger auxiliary
Jastrow networks are deferred.
