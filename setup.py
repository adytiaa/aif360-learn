from setuptools import setup, find_packages

long_description = """The AI Fairness 360 toolkit is an open-source
library to help detect and remove bias in machine learning models. The
AI Fairness 360 Python package includes a comprehensive set of metrics
for datasets and models to test for biases, explanations for these
metrics, and algorithms to mitigate bias in datasets and models.

We have developed the package with extensibility in mind. This library
is still in development. We encourage the contribution of your
datasets, metrics, explainers, and debiasing algorithms."""

setup(name='aif360-learn',
      version='0.1.dev1',
      description='AI Fairness 360 Learn',
      author='aif360 [learn] developers',
      author_email='adrin.jalali@gmail.com',
      url='https://github.com/aif360-learn/aif360-learn',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache License 2.0',
      packages=find_packages(),
      python_requires='>=3.5',
      setup_requires=['numpy'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas==0.23.3',
          'scikit-learn',
          'numba',
          'lime',
      ],
      extras_require={
          'tests': [
              'nbformat',
              'nbconvert',
              'jupyter_client',
              'ipykernel',
              'ipython',
              'ipywidgets',
          ],
          'docs': [
              'nbformat',
              'nbconvert',
              'jupyter_client',
              'ipykernel',
              'ipython',
              'ipywidgets',
              'tqdm',
              'matplotlib',
              'pytest',
              'cvxpy>=1.0',
              'numba',
              'tensorflow==1.1.0',
              'networkx==1.11',
              'BlackBoxAuditing',
          ]
      },
      package_data={'aiflearn': ['data/*', 'data/*/*', 'data/*/*/*']},
      include_package_data=True,
      zip_safe=False)
