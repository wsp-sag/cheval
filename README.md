# Cheval (wsp-cheval)

[![Conda Latest Release](https://anaconda.org/wsp_sap/wsp-cheval/badges/version.svg)](https://anaconda.org/wsp_sap/wsp-cheval)
[![Conda Last Updated](https://anaconda.org/wsp_sap/wsp-cheval/badges/latest_release_date.svg)](https://anaconda.org/wsp_sap/wsp-cheval)
[![Platforms](https://anaconda.org/wsp_sap/wsp-cheval/badges/platforms.svg)](https://anaconda.org/wsp_sap/wsp-cheval)
[![License](https://anaconda.org/wsp_sap/wsp-cheval/badges/license.svg)](https://github.com/wsp-sag/cheval/blob/master/LICENSE)

Cheval is a Python package for high-performance evaluation of discrete-choice (logit) models. It's largely built upon the Pandas, NumPy, and NumExpr packages; along with some custom Numba code for performance-critical bottlenecks.

The name is an acronym for "CHoice EVALuator" but has a double-meaning as _cheval_ is the French word for "horse" - and this package has a lot of horsepower! It has been designed for use in travel demand modelling, specifically microsimulated discrete choice models that need to process hundreds of thousands of records through a logit model. It also supports "stochastic" models, where the probabilities are the key outputs.

Cheval is owned and published by WSP Canada's System Analytics for Policy group.

## Key features

Cheval contains two main components:

- `cheval.ChoiceModel` which is the main entry point for discrete choice modelling
- `cheval.LinkedDataFrame` which helps to simplify complex utility calculations.

These components can be used together or separately.

Cheval is compatible with Python 3.6+

## Installation

### With `conda`

Cheval is hosted on the wsp_sap conda channel and can be installed with conda by running the following command:

```batch
conda install -c wsp_sap wsp-cheval
```

#### Build and install locally

You can also choose to install Cheval with conda locally. This option requires you first build the conda package to your computer's a local Conda channel. This can be done by using [conda-build](https://github.com/conda/conda-build), which will set up a local Conda channel (i.e. `conda-bld`) in your Conda installation folder. The following code block provides the commands to build and install Cheval locally using Conda.

```batch
(base) C:\> conda build "<path to local cheval repository folder>/conda_recipe"
(base) C:\> conda install -c local wsp-cheval
```

### With `pip`

Currently, Cheval is not hosted on PyPI. The best way to install cheval with `pip` is to link directly to this GitHub repo:

```batch
pip install git+https://github.com/wsp-sag/cheval.git
```

This requires you to download and install a [standalone Git client](https://git-scm.com/downloads) to communicate with GitHub.

> **Windows Users:** It is recommended to install Cheval from inside an activated Conda environment. Cheval uses several packages (NumPy, Pandas, etc.) that will otherwise not install correctly from `pip` otherwise. For example:

```batch
C:\> conda activate base
(base) C:\> pip install git+https://github.com/wsp-sag/cheval.git
```
