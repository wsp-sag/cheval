from setuptools import find_packages, setup

import re
VERSION_FILE = "cheval/_version.py"
version_line = open(VERSION_FILE, "rt").read()
regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(regex, version_line, re.M)
if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSION_FILE,))

setup(
    name='cheval',
    version=version_string,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'pandas>= 0.22, <0.24',
        'numpy>=1.14',
        'astor',
        'numba',
        'numexpr3',
        'attrs',
        'deprecated'
    ]
)
