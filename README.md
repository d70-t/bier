Basic Illumination Estimation and Regression
============================================

This is a fast analytic model which estimates radiance at the earth suface using only few parameters characterizing the atmosphere.

The model is based on Gregg and Carder (1990) and the Water Colour Simulator [WASI](ftp://ftp.dfd.dlr.de/pub/WASI/).

Currently, this repository is more a snaphot of previous work and highly unorganized, but should work as a starting point for further development.

How it (should) work
--------------------

The model is a very simplistic analytic representation of the scattering process in the atmosphere.
It could be undestood as an about 1.5 fold scattering model as one scattering and the remaining extinction from futher scatterings is considered.
This simplification allows to fit the model to measured radiance data quite fast and thereby obtain unknown atmospheric parameters (like angstrom exponent or turbidity).

### For users

The typical application is that some radiance data has been measured and the model should be fitted to it.
This is done by the ``bier/runFit.py`` script, which is configured by a [YAML](http://yaml.org/) file.
Examples can be found in the ``tasks`` directory.
This will generary a numpy ``.npy`` file and a ``.prop.yaml`` file of results.
The numpy file contains results of big arrays, while the YAML file contains metadata and scalar results.
There is another script (``createOutputPreviews.py``) which can be used to generate a short PDF report of the fit result.
Examples of how to call these scripts can be found in ``results/Makefile``.

### For developers

#### Working principle

The model is written in terms of [sympy](http://www.sympy.org) expressions, which are compiled into [theano](http://www.deeplearning.net/software/theano/) functions.
Examples can be found in ``bier/downwelling.py``.

This model is set up by a ``fitter``, which subsequently fits measurement data to the model.
There are multiple implementations of the fitter, depending on how to fit (global or pointwise), if it should be run in parallel and how (using zeromq or celery).
For each of the fitter implementations, the user can select the fitting method which will be used for the actual fit (e.g. Nelder-Mead, TNC, basinhopping etc...).

#### Coding

Sadly the code is not in a very well shape, however this is supposed to change...

First, patches are always welcome so feel free to submit a lot of them!

Idelly, at some point the code should converge to [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines so for new development, it is strongly encuraged to follow this guideline.
This also includes renaming functions such that the naming conventions can be fulfilled.
A notable exception to naming conventions will be the written analytic models which should follow the respective papers as close as possible and due to the mathematical nature, naming conventions might be more a burden than a help.

