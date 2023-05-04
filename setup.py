from setuptools import find_packages, setup

import versioneer

setup(
    name='wsp-cheval',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='High-performance discrete-choice (logit) travel demand model evaluation',
    url='https://github.com/wsp-sag/cheval',
    author='WSP',
    maintainer='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    install_requires=[
        'pandas>=0.24,<1.5',
        'numpy>=1.20',
        'astor',
        'numba>=0.48',
        'numexpr',
        'attrs>=19.3'
    ],
    python_requires='>=3.7'
)
