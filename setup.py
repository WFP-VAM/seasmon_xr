#!/usr/bin/env python

from setuptools import find_packages, setup

# requirements = [ ]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="",
    author_email="WFP-VAM",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="xarray tools for seasonal monitor",
    # install_requires=requirements,
    license="MIT license",
    long_description="",
    # include_package_data=True,
    keywords="seasmon_xr",
    name="seasmon_xr",
    packages=find_packages(include=["seasmon_xr", "seasmon_xr.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/WFP-VAM/",
    version="0.1.0",
    zip_safe=False,
)
