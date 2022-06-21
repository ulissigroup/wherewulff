"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="wherewulff",
    version="0.0.1",
    description="WhereWulff: An Automated Workflow to Democratize and Scale Complex Material Discovery for Electrocatalysis.",
    url="https://github.com/ulissigroup/mo-wulff-workflow",
    packages=find_packages(),
    include_package_data=True,
)