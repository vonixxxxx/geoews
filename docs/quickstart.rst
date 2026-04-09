Quickstart
==========

.. code-block:: python

   import numpy as np
   from geoews import ManifoldEWS

   x = np.random.default_rng(42).standard_normal(1000)
   result = ManifoldEWS(window=50, cumul_window=30).fit(x).detect()
   print(result.alert_index, result.threshold)
