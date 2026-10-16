Tiny fixed-orbital MDJ native fixture
====================================

``wftrain_mdj.xml`` reuses the exact tiny H4-derived DETS/J2 model from
``test_LegacyWaveFunctionAdapter.cpp``. The hydrogen STO-3G contractions and
fixed four-orbital coefficient matrix originate in ``../H4_ae/H4.wfs.xml``
(SHA-256 ``5413985d750478e83e04cb35f1285ed6d1bbb12a3c8ae798268999e98cea6585``).
Linear test geometry, electron positions and small four-coefficient J2 functions
are explicit correctness initializations. The first CI coefficient0.6 is fixed;
the second starts exactly0. Eight J2 coefficients are active jointly with CI1.

The shared checker derives the active manifest from the XML instead of assuming
the H2 parameter count. It checks canonical updates, fixed SPO/CI0 export,
inference accepted-version labels, exact inference isolation, and split restart.
Example from this directory:

.. code-block:: text

   python3 ../H2_ae_conventional_wftrain/check_sj_training.py /absolute/qmcpack \
     wftrain_mdj.xml --companion-optimizer sr --restart

This runs representative Adam and bounded SR,2updates with4samples/update,
plus Adam inference isolation and split continuation. SR has damping0.01 and
Krylov caps32iterations,128operator applications,10seconds, one damping attempt.
``--ci-representation csf`` expresses the same two terms as single-determinant
CSFs; ``--profile-only`` can select one bounded Adam case for that representation.
No external orbital data or trained model is used, and no convergence or orbital
rotation/advanced-linear-optimizer qualification is implied.
