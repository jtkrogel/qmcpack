Native parameter-only paired initialization
===========================================

An explicit paired internal initializer can replace its initial draws with
canonical real parameters from ONE exact native WFTrain root envelope::

  <psiformer name="pf" source="ion0" system="all_electron"
      initialization="psiformer_complex_periodic_v1" initialization_seed="17"
      feature_policy="periodic_torus_v1" optimize="no" precision="fp64_reference"
      gpu="cuda" max_clone_numeric_bytes="16777216"
      max_gpu_numeric_bytes="67108864"
      parameter_checkpoint="checkpoint.manifest.h5.generation-TOKEN.root.h5"/>

Use the exact selected root filename from a completed native checkpoint family,
NOT its manifest, sampler shard, glob or guessed latest generation. The current
component name/provider and complete canonical schema must match the archive;
e.g. saved provider psiformer/pf requires name="pf". Root completion cookie,
schema/count/values/finiteness/fingerprint/source-version agreement are checked.
No external parameters/configuration/export attributes may be combined with the
internal paired initializer. This attribute is not supported by the legacy REAL
initialization path. CPU paired inference also supports this initialization;
select gpu="no" and omit max_gpu_numeric_bytes for same-parameter comparisons.

This is PARAMETER-ONLY WARMSTART, NOT restart or reconstruction of the saved
physical wavefunction. CURRENT XML ions, geometry, cell, masses and twist apply;
deliberate transfer to another twist is allowed. Native roots do not contain
recoverable separate physical metadata. Saved source version is provenance;
runtime version starts freshly and advances through ordinary initialization.
Optimizer/progress/walkers/RNG are NOT restored, and source files are not changed.

Temporary host canonical values/schema during import are initialization I/O,
outside numerical owner/clone caps; these caps are not process-peak memory limits.
Immutable GPU image construction follows import, with ordinary clone/image sharing
and no CPU numerical tapes on GPU clones. Runtime GPU publication, GPU training,
parameter derivatives and checkpoint restart remain unsupported.
