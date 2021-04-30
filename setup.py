from os import path
from pkg_resources import safe_version
from setuptools import find_packages, setup

version = {}
with open(path.join(path.dirname(path.realpath(__file__)), 'cheval', 'version.py')) as fp:
    exec(fp.read(), {}, version)

setup(
    name='wsp-cheval',
    version=safe_version(version['__version__']),
    description='High-performance discrete-choice (logit) travel demand model evaluation',
    url='https://github.com/wsp-sag/cheval',
    author='WSP, Peter Kucirek',
    maintainer='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    install_requires=[
        'pandas>=0.22',
        'numpy>=1.14',
        'astor',
        'numba>=0.45',
        'numexpr',
        'deprecated',
        'attrs>=19.3'
    ],
    python_requires='>=3.6'
)
