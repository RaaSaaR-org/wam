"""Estimator pairs for PR-08 §4 — RGB in, an object mask and a depth map out.

A module in this package implements the contract ``scripts/measure_est_drift.py`` fixes in its
``Estimators`` class, and nothing else::

    segment(rgb: np.ndarray) -> np.ndarray         # (H, W) bool, the object in THIS frame
    estimate_depth(rgb: np.ndarray) -> np.ndarray  # (H, W) float32, METRES

and may declare ``ESTIMATOR_NAME``, ``ESTIMATOR_VERSION`` and ``GATE_QUALIFIED``. The last one is
read with a default of ``False``: an estimator is a gate input only if its author said so in
writing, in the module, where a reviewer can find it. That is why this is a package of named
modules rather than a flag on a script — the artifact has to be able to say *which* pair produced
the number, because PR-08 §4 step 2 requires the segmenter here to be the SAME one ``GEOM_TOL`` was
measured with, and a budget measured with one estimator and applied with another is a subtraction
of two different quantities that nothing downstream would notice.

Importing this package imports no model, no torch and no weights. Each estimator module is
responsible for its own lazy loading, so that ``--estimators estimators.<name>`` can fail by NAME
before anything is fetched.
"""
