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

The MSE regression is CPU only and does not itself qualify ENERGY optimization, GPU MSE,
gauge optimization, general SPO targets, the deferred SCI ansatz or a different
complex representation. The old real orbital-MSE regression remains separate.

The optional ``--energy-reference-profile`` mode derives a CPU ENERGY pilot
from the same Ne2/twist/cell/internal initializer. It uses Adam1e-7, W2/blocks2
(four samples), reference search with ESS fraction .5, maximum weight .5 and
confidence2, preserving the 512MiB retained cap and inference settings.
It runs only two iterations in one 60-second GNUtime/RSS-recorded process;
full/split follow-ons require separate review. Six finite candidate receipts and
natural selected-alpha/version rules are checked without forcing a commit.
For an initial rejection, version0 and zero moments/count are checked; initial
accepted parameter equality is not independently observable through this XML.
Later rejections compare exact previous values and optimizer state. Existing
isolation units supply candidate ownership proof, not this native pilot.

After the separately qualified pilot, ``--energy-reference-restart`` reuses the
same settings for continuous2/split1/resume2 under a 120-second aggregate limit.
It compares model values, Adam state/count, sampler/RNG, nonclock scientific
history and physical inference exactly, and checks one early CPU endpoints
rejection. Initial tensor equality remains unobserved; any imaginary-head change
reported here compares first and second accepted exports, not initialization.

The CPU ENERGY/reference pilot and complete restart workflow both passed at
application source ``9f94ee23c`` (SHA256
``958b8fdb33eb0e46f18977c72a936d6c73158fcaa3543b2205c10932d24cf306``).
The pilot took 7.77 seconds including startup and 826612KiB maximum RSS,
with no cap increase. The full workflow selected alpha1 naturally for both
continuous updates and passed exact restart, physical inference and CPU endpoints
rejection. Imaginary-head change was observed between first and second exports.
Initial tensor equality and alpha0 assertions were not exercised by these natural
decisions. ``deterministic-H2-psiformer-complex-cpu-reference-restart`` registers
only this CPU/full-precision profile; no GPU, endpoints or general selection claim.
