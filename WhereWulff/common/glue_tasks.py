"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from fireworks import explicit_serialize, FiretaskBase
from monty.shutil import gzip_dir


@explicit_serialize
class GzipPrevDir(FiretaskBase):
    """
    Task to gzip a specific directory through calc_dir.
    """

    required_params = ["calc_dir"]
    optional_params = []

    def run_task(self, fw_spec=None):
        cwd = self["calc_dir"]
        gzip_dir(cwd)
