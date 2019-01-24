#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone aif360-learn/aif360-learn repository in to a local repository.
# We use a cached directory with three aif360-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

if [ $TRAVIS_OS_NAME = "linux" ]
then
	export CC=/usr/lib/ccache/gcc
	export CXX=/usr/lib/ccache/g++
	# Useful for debugging how ccache is used
	# export CCACHE_LOGFILE=/tmp/ccache.log
	# ~60M is used by .ccache when compiling from scratch at the time of writing
	ccache --max-size 100M --show-stats
fi

make_conda() {
	TO_INSTALL="$@"
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    # If Travvis has language=generic, deactivate does not exist. `|| :` will pass.
    deactivate || :

    # Install miniconda
    if [ $TRAVIS_OS_NAME = "osx" ]
	then
		fname=Miniconda3-latest-MacOSX-x86_64.sh
	else
		fname=Miniconda3-latest-Linux-x86_64.sh
	fi
    wget https://repo.continuum.io/miniconda/$fname \
        -O miniconda.sh
    MINICONDA_PATH=$HOME/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
    export PATH=$MINICONDA_PATH/bin:$PATH
    conda update --yes conda

    conda create -n testenv --yes $TO_INSTALL
    source activate testenv
}

if [[ "$DISTRIB" == "conda" ]]; then
    TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov numpy scipy"
    make_conda $TO_INSTALL
    #pip install numpy
    #pip install -r requirements.txt
fi

pip install coverage codecov
pip install sphinx numpydoc  # numpydoc requires sphinx


# Build aif360-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "\
try:
    import pandas
    print('pandas %s' % pandas.__version__)
except ImportError:
    pass
"
# python setup.py develop
pip install -e.[docs,docs]
if [ $TRAVIS_OS_NAME = "linux" ]
then
	ccache --show-stats
fi
# Useful for debugging how ccache is used
# cat $CCACHE_LOGFILE
