Modern estimator and DMC application qualification
===================================================

``check_modern_integration.py`` derives tiny initialized-model application
inputs from an existing complete physical XML fixture. It records the binary
and generated input SHA256, MPI rank count, and relevant GPU crowd/tile settings.
Each workflow runs two modern driver sections and requires finite scalar output.
Each invocation preserves a fresh ``run_*`` child under its output directory,
so repeated CTest runs retain earlier evidence without deleting output.

Estimator mode adds native MomentumDistribution and OneBodyDensityMatrices
estimators. The density matrix projection uses an independently supplied
three-orbital free-particle SPO set. It does not require an orbital API from
the many-electron component. Both sections must produce finite, nonzero momentum
and spin-resolved density-matrix HDF observations. The verifier uses h5py when
installed or the existing h5dump executable on PATH.

For example, from a configured environment with the production binary::

  python3 check_modern_integration.py --binary /path/to/qmcpack_complex \
    --fixture complex_twist.xml --mode estimators --output /scratch/estimator_probe

DMC mode uses native PBYP drift diffusion, dynamic population control, native
gradient/Laplacian checks and nonlocalmoves=no. Choose an existing qualified AE
or scalar-ECP locality fixture. A longer tiny step sequence can exercise real
population changes; inspect NumOfWalkers and AvgSentWalkers in the DMC output
before claiming branching or MPI migration coverage::

  python3 check_modern_integration.py --binary /path/to/qmcpack_complex \
    --fixture complex_twist.xml --mode dmc --timestep .02 --steps 12 \
    --mpi-ranks 2 --restart --output /scratch/dmc_probe

``--restart`` reloads the native walker/RNG checkpoint into a new application
process and verifies two additional sections. This checks reconstruction and
resumed sampling; it does not assert exact uninterrupted DMC branch-state replay.
The native GL debug tolerance is unchanged. Independent strict FP64 wavefunction
product tests provide the separate numerical oracle.

Selected mode uses n_electron, electrons_per_move=1, cyclic selection, and two
VMC sections. It checks parsed application dispatch, complete finite physical
observations and resumed sampling. It does not claim exact cyclic VMC file
continuation.

For GPU-backed components use --gpu cuda or --gpu hip with the configured native
tile, active-crowd and aggregate-device admission settings. Estimators remain
CPU-side consumers. Scheduler MPI users can choose --mpi-launcher srun from an
allocated node; this constrains child steps to that exact node. Allocate one GPU
per MPI rank where applicable. T-move propagation is outside this qualification.
