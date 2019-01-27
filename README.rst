.. -*- mode: rst -*-

|Documentation|_ |Travis|_ |Codecov|_ |CircleCI|_ |Python35|_

.. |Documentation| image:: https://readthedocs.org/projects/aif360-learn/badge/?version=master
.. _Documentation: https://aif360-learn.readthedocs.io/en/master/?badge=master

.. |Travis| image:: https://api.travis-ci.org/aif360-learn/aif360-learn.svg?branch=master
.. _Travis: https://travis-ci.org/aif360-learn/aif360-learn

.. |Codecov| image:: https://codecov.io/github/aif360-learn/aif360-learn/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/aif360-learn/aif360-learn?branch=master

.. |CircleCI| image:: https://circleci.com/gh/aif360-learn/aif360-learn/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/aif360-learn/aif360-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/aif360-learn


AI Fairness 360 Learn
=====================

The AI Fairness 360 Learn started as a fork of IBM/AIF360 toolkit which is an
open-source library to help detect and remove bias in machine learning models.
This project focuses on making *aif360* compatible with *scikit-learn*, and
hence called *aif360-learn*.

There are two potential approaches in order to have a *scikit-learn* compatible
library:

- A wrapper library which depends on AIF360.
- A fork of AIF360 with necessary changes applied to it.

This fork is taking the latter approach for the following reasons:

- Compatibility with the upstream (python and dependencies), and dropped
  Python2 support.
- Have the related communication on public channels
- Encourage researchers and the community to include more algorithms in the
  library and help them on the way for their methods to better reach a wider
  community

Hence, this project started as a fork of `IBM/AIF360
<https://github.com/IBM/AIF360>`_, and will slowly move further away from
it. As a result, most of the content is what you see on the original
repository, for a while.

Get in touch with us on `Gitter <https://gitter.im/aif360-learn/community>`_!

The AI Fairness 360 Python package includes a comprehensive set of metrics for
datasets and models to test for biases, explanations for these metrics, and
algorithms to mitigate bias in datasets and models.

The `AI Fairness 360 interactive experience
<http://aif360.mybluemix.net/data>`_ provides a gentle introduction to the
concepts and capabilities. The `tutorials and other notebooks
<https://github.com/IBM/AIF360/tree/master/examples>`_ offer a
deeper, data scientist-oriented introduction. The complete API is also
available.

Being a comprehensive set of capabilities, it may be confusing to figure out
which metrics and algorithms are most appropriate for a given use case. To
help, we have created some `guidance material
<http://aif360.mybluemix.net/resources#guidance>`_ that can be consulted.

We have developed the package with extensibility in mind. This library is still
in development. We encourage the contribution of your metrics, explainers, and
debiasing algorithms.

Supported bias mitigation algorithms
------------------------------------

- Optimized Preprocessing (`Calmon et al., 2017
  <http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention>`_)
- Disparate Impact Remover (`Feldman et al., 2015
  <https://doi.org/10.1145/2783258.2783311>`_)
- Equalized Odds Postprocessing (`Hardt et al., 2016
  <https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning>`_)
- Reweighing (`Kamiran and Calders, 2012
  <http://doi.org/10.1007/s10115-011-0463-8>`_)
- Reject Option Classification (`Kamiran et al., 2012
  <https://doi.org/10.1109/ICDM.2012.45>`_)
- Prejudice Remover Regularizer (`Kamishima et al., 2012
  <https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3>`_)
- Calibrated Equalized Odds Postprocessing (`Pleiss et al., 2017
  <https://papers.nips.cc/paper/7151-on-fairness-and-calibration>`_)
- Learning Fair Representations (`Zemel et al., 2013
  <http://proceedings.mlr.press/v28/zemel13.html>`_)
- Adversarial Debiasing (`Zhang et al., 2018
  <http://www.aies-conference.com/wp-content/papers/main/AIES_2018_paper_162.pdf>`_)
- Meta-Algorithm for Fair Classification (`Celis et al.. 2018
  <https://arxiv.org/abs/1806.06055>`_)

Supported fairness metrics
--------------------------

- Comprehensive set of group fairness metrics derived from selection rates and
  error rates
- Comprehensive set of sample distortion metrics
- Generalized Entropy Index (`Speicher et al., 2018
  <https://doi.org/10.1145/3219819.3220046>`_)


Setup
-----

Requirements:

- Python 3.5+

Installation is easiest on a Unix-like system running Python 3. See the
`Troubleshooting`_ section if you have issues with other
configurations.

(Optional) Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package requires specific versions of many Python packages which may
conflict with other projects on your system. A virtual environment manager is
strongly recommended to ensure dependencies may be installed safely. If you
have trouble installing AIF360, try this first.

Conda
"""""

Conda is recommended for all configurations though Virtualenv is generally
interchangeable for our purposes `CVXPY`_ may require conda in some
cases). Miniconda is sufficient (see `the difference between Anaconda and
Miniconda
<https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda>`_
if you are curious) and can be installed from `here
<https://conda.io/miniconda.html>`_ if you do not already have it.

Then, to create a new Python 3.5 environment, run::


    conda create --name aiflearn python=3.5
    conda activate aiflearn


The shell should now look like `(aiflearn) $`. To deactivate the environment,
run::


    (aiflearn)$ conda deactivate


The prompt will return to `$`.

Note: Older versions of conda may use `source activate aiflearn` and `source
deactivate` (`activate aiflearn` and `deactivate` on Windows).

Install with minimal dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version from PyPI, run: ::


    pip install aif360-learn


This package supports Python 3.5+.

Some algorithms require additional dependencies not included in the minimal
installation. To use these, we recommend a full installation.

### Full installation

Clone the latest version of this repository::


    git clone https://github.com/aif360-learn/aif360-learn


If you'd like to run the examples, download the datasets now and place them in
their respective folders as described in
`aiflearn/data/README.md <aiflearn/data/README.md>`_.

Then, navigate to the root directory of the project and run::


    pip install .


Run the Examples
""""""""""""""""

To run the example notebooks, install the additional requirements as follows::


    pip install -r requirements.txt


Then, follow the `Getting Started <https://pytorch.org>`_ instructions from
PyTorch to download and install the latest version for your machine.

Finally, if you did not already, download the datasets as described in
`aiflearn/data/README.md <aiflearn/data/README.md>`_ but place them **in the
appropriate sub-folder** in
`$ANACONDA_PATH/envs/aiflearn/lib/python3.5/site-packages/aif360/data/raw`
where `$ANACONDA_PATH` is the base path to your conda installation (e.g.
`~/anaconda`).

Troubleshooting
^^^^^^^^^^^^^^^

If you encounter any errors during the installation process, look for your
issue here and try the solutions.

TensorFlow
""""""""""

In some cases, the URL is required for installation::


    # WINDOWS
    pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl

    # MACOS
    pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl

    # LINUX
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl


Substitute Python version numbers for your configuration as appropriate (Note:
TensorFlow 1.1.0 only supports Python 3.5 officially on Windows).

TensorFlow is only required for use with the
`aiflearn.algorithms.inprocessing.AdversarialDebiasing` class.

CVXPY
"""""

On Windows, you may need to download the appropriate `Visual Studio C++
compiler for Python <https://wiki.python.org/moin/WindowsCompilers>`_. Then,
re-run::


    pip install -r requirements.txt


See the `CVXPY Installation
Instructions <https://www.cvxpy.org/install/index.html#windows>`_
for an alternate installation procedure using conda.

CVXPY is only required for use with the
`aiflearn.algorithms.preprocessing.OptimPreproc` class.


Using aiflearn
--------------

The `examples` directory contains a diverse collection of jupyter notebooks
that use this package in various ways. Both tutorials and demos illustrate
working code using `aiflearn`. Tutorials provide additional discussion that
walks the user through the various steps of the notebook. See the details about
`tutorials and demos here <examples/README.md>`_.

Citing AIF360
-------------

A technical description of AI Fairness 360 is available in this
`paper <https://arxiv.org/abs/1810.01943>`_. Below is the bibtex entry for this
paper.::


    @misc{aif360-oct-2018,
        title = "{AI Fairness} 360:  An Extensible Toolkit for Detecting,
                  Understanding, and Mitigating Unwanted Algorithmic Bias",
        author = {Rachel K. E. Bellamy and Kuntal Dey and Michael Hind and
        Samuel C. Hoffman and Stephanie Houde and Kalapriya Kannan and
        Pranay Lohia and Jacquelyn Martino and Sameep Mehta and
        Aleksandra Mojsilovic and Seema Nagar and Karthikeyan Natesan
        Ramamurthy and John Richards and Diptikalyan Saha and Prasanna
        Sattigeri and Moninder Singh and Kush R. Varshney and Yunfeng Zhang},
        month = oct,
        year = {2018},
        url = {https://arxiv.org/abs/1810.01943}
    }


AIF360 Videos
-------------

- Introductory `video <https://www.youtube.com/watch?v=X1NsrcaRQTE>`_ to AI
  Fairness 360 by Kush Varshney, September 20, 2018 (32 mins)
