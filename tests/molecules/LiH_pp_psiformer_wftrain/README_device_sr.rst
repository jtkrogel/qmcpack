Ordinary-locality BFD DEVICE SR workflow
======================================

``deterministic-LiH-pp-psiformer-device-sr-restart`` uses the original internally
initialized LiH BFD model: Ne2, all1,593,857 parameters trainable, seed37,
sampler seed91427, two walkers, one crowd/block/step, alle/no drift, timestep1e-4.
The original learning rate1e-10 is retained; fixed positive damping.3 and SR
reuse=on use explicit256MiB DEVICE/1MiB HOST solver caps,1GiB retained cap and
2GiB aggregate DEVICE/256MiB external reserve. No external trained model or
new fixture generator is required. Only CUDA real FP64 with Python h5py/numpy
registers this test; existing GPU launcher/resource scheduling is reused.

The helper first runs two continuous updates, then reuses that protected result
for split1/resume2. It checks actual complete ECP append, positive solver work,
committed versions, finite all-P exports, cold mirror ordering, exact ROOT and
sampler envelopes/all parameter datasets/non-timing history, protected inputs
and an actual trainable parameter-bit change between accepted versions1 and2.
There are no frozen parameters in this native profile. Each child is capped
at60s, the helper at180s, CTest at200s. Repeated runs retain unique subdirectories
under ``--output-parent``; manual ``--output`` still requires a new directory.

Actual case80 passed at source d52833aff4259baa57799c46d21ab4ac043daaa6,
app SHA2567da17545a7c8b7d52af09be59819f7209603c21684055350fffdc5c9d0398e56,
task helper9280084711fd19f40c72b7571b253c691ee4090e816b7b14e6512a3ae96d8c3b.
Continuous2 took2.820s/RSS741712KiB; split/resume2.469s each. Exact initialized
restart passed without rerunning the pilot or changing physics/controls.
This is a bounded workflow/memory correctness qualification, NOT convergence
or optimization performance. HOST SR ECP, DLA/spin-orbit, complex, MPI/crowd
expansion and altered optimizer controls remain outside this profile.

Prerequisites were the separate two-configuration low-level complete-ECP SR
oracle (case76) and component24842assertion proof (case78). The first rejected
component review and case77 capture-descriptor-only failure are preserved in
Task41 evidence. Registration is tests-only; its discovery/path checks do not
claim another scientific rerun of the final registration source.
