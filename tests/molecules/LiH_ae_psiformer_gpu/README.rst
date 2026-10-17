Experimental resident DEVICE Adam authority
===========================================

The explicit environment opt-in ``PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY=1``
enables the bounded native DEVICE-authoritative ENERGY Adam route. It is off by
default; resident byte budgets alone do not change the existing HOST route.
The qualified profile is REAL/open FP64 all-electron Coulomb, one PsiFormer
factor, MPI one, one crowd/block/step, any positive walker count, DEVICE
``move="alle"`` without drift, and nonserializing FULL refresh. Line search,
selection/scaling/decay/norm controls, inference/debug GL, hard plans, ECP,
complex and multi-GPU authority remain outside this admission.

This does not remove HOST storage. Warm proposals/publication retain owning
DEVICE candidates and moments without parameter-vector HOST round trips.
Default CPU readers fence a stale canonical/graph-leaf mirror once per version.
Initial load, restart, export and checkpoint are explicit cold HOST boundaries;
checkpoint moment synchronization and existing file formats are preserved.

Bounded qualification recipe
----------------------------

Use a real CUDA build. From its wavefunction-test working directory, export the
initial protected tiny REAL LiH model through the numerical authority gate::

  PSIFORMER_TEST_RESIDENT_MODEL_EXPORT=/absolute/new/tiny-real-lih.h5 \
    ./test_psiformer_streaming_derivative '[gpu_resident_authority]'

The gate exports before mutation. Its P666 model has Ne4=2u2d, D2, width8,
heads2, one attention block and seed37. Its AE ion charges/geometry match
``input.xml``; do not substitute a paired or pseudopotential model.
Then run the finite native checker with the matching application/source::

  python3 SOURCE/tests/molecules/LiH_ae_psiformer_gpu/check_resident_device_authority.py \
    /absolute/build/bin/qmcpack --source /absolute/SOURCE \
    --model /absolute/new/tiny-real-lih.h5 --output /absolute/new/native-check

The checker sets the opt-in and required explicit budgets, uses W2, synchronous
status checks and a 3 GiB aggregate admission cap (not an availability promise).
It requires three updates, cold export/checkpoint equality, exact split restart,
strict same-sample HOST Adam parity and source-accounted full-window transfers.
Small synchronous status/spatial traffic totals 7060 D2H bytes for this fixture;
an aggregate D2H<8P heuristic is invalid. Exact accounting rejects any additional
P-sized download; the only canonical mirror occurs afterward at the cold final
checkpoint. This is a correctness/transfer check, not a throughput benchmark.
