# Cheval (wsp-cheval)

Cheval is a Python package for high-performance evaluation of discrete-choice (logit) models. It's largely built upon the Pandas, NumPy, and NumExpr packages; along with some custom Numba code for performance-critical bottlenecks.

The name is an acronym for "CHoice EVALuator" but has a double-meaning as _cheval_ is the French word for "horse" - and this package has a lot of horsepower! It has been designed for use in travel demand modelling, specifically microsimulated discrete choice models that need to process hundreds of thousands of records through a logit model. It also supports "stochastic" models, where the probabilities are the key outputs.

Cheval is owned and published by WSP Canada's System Analytics for Policy group.

## Key features

Cheval contains two main components:

- `cheval.ChoiceModel` which is the main entry point for discrete choice modelling
- `cheval.LinkedDataFrame` which helps to simplify complex utility calculations.

These components can be used together or separately.

Cheval is compatible with Python 3.6+

## Current Status

*Version 0.1.1* is the latest release of cheval.

## Installation

### With `pip`

Currently, Cheval is not yet hosted on PyPI, Conda, or other distribution services. The best way to install Balsa is using `pip` to install directly from GitHub:

```batch
pip install git+https://github.com/wsp-sag/cheval.git
```

Git will prompt you to login to your account (also works with 2FA) before installing. This requires you to download and install a [standalone Git client](https://git-scm.com/downloads) to communicate with GitHub.

> **Windows Users:** It is recommended to install Cheval from inside an activated Conda environment. Cheval uses several packages (NumPy, Pandas, etc.) that will otherwise not install correctly from `pip` otherwise. For example:

```batch
C:\> conda activate base

(base) C:\> pip install git+https://github.com/wsp-sag/cheval.git
...
```

### With `conda`

Cheval can be installed with Conda, but requires you to install it from a local Conda channel. This can be done by using [conda-build](https://github.com/conda/conda-build), which will create a Conda package for Cheval (that has been cloned from GitHub onto your machine) and set up a local Conda channel (i.e. `conda-bld`) in your Conda installation folder. conda-build must be installed in your base Conda environment. Once the Conda package is built, you can install it to your Conda environment of choice using `conda install`.

The following code block provides the commands to install Balsa using Conda.

```batch
(base) C:\> conda build "<path to local cheval repository folder>/conda_recipe"

...

(base) C:\> conda install -c "<path to your conda installation folder>/conda-bld" wsp-cheval
```
