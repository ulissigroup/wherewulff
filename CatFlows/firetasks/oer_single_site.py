import uuid
import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE

# OER single site WF ?

@explicit_serialize
class OERSingleSiteFireTask(FiretaskBase):
    """
    OER Single Site FireTask.

    Args:

    Returns:
        OERSingleSite Firetaks.
    """

    required_params = ["reduced_formula", "slab", "adsorbates", "vasp_cmd", "db_file"]
    optional_params = []

    def run_task(self, fw_spec):

        # Variables
        reduced_formula = self["reduced_formula"]
        slab = self["slab"]
        miller_index = self["miller_index"]
        adsorbates = self["adsorbates"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Get PBX collection from DB
        pbx_collection = mmdb.db[f"{reduced_formula}-{miller_index}_surface_pbx"]

        # From collection retrieve/decide the most stable surface termination OH/Ox

        # Reconstruct the PMG Slab object for the selected termination

        # Generate OER single site intermediates


        return FWAction()

