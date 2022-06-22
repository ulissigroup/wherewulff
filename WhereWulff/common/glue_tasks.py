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
