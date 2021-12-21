"""
#
# ensemble-clustering setup.py
#
# Copyright(c) 2021, Antoine Emil Zambelli.
#
"""

from setuptools import setup, find_packages


version = '0.0.3'

setup(
    name='ensemble-clustering',
    version=version,
    url='https://github.com/antoinezambelli/ensemble-clustering',
    license='MIT',
    author='Antoine Emil Zambelli',
    author_email='antoine.zambelli@gmail.com',
    description='Ensemble Clustering',
    long_description='Code companion to: Ensemble Method for Cluster Number Determination and Algorithm Selection in Unsupervised Learning',
    packages=find_packages(
        exclude=(
           '.*',
           'EGG-INFO',
           '*.egg-info',
           '_trial*',
           "*.tests",
           "*.tests.*",
           "tests.*",
           "tests",
           "examples.*",
           "examples",
        )
    ),
    include_package_data=True,
    install_requires=[
        'fastcluster>=1.2.4',
        'numpy>=1.20.3',
        'scikit-learn>=1.0',
        'scipy>=1.7.1',
        'tqdm>=4.62.3',
    ]
)
