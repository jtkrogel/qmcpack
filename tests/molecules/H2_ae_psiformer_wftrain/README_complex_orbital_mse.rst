CPU complex orbital-MSE smoke test
==================================

``wftrain_complex_orbital_mse.xml`` uses the existing paired periodic profile
and a frozen actual FreeOrbital single-reference teacher, with a fixed nonzero
twist and gauge. The student fits the periodic orbital part, not the teacher's
determinant amplitude. No external assets are needed.

The checker requires two committed Adam updates, a changed imaginary head,
scheduled physical inference and exact continuous versus split/resumed model,
optimizer and sampler/RNG payloads. Scientific history and physical inference
columns are exact; only the BlockCPU timing column is excluded. Changed teacher
twist and changed-cell restart are rejected without modifying the checkpoint.
The checker enforces 120 seconds aggregate and 45 seconds per native child.

This regression is CPU only. It does not qualify ENERGY optimization, GPU MSE,
gauge optimization, general SPO targets, the deferred SCI ansatz or a different
complex representation. The old real orbital-MSE regression remains separate.
