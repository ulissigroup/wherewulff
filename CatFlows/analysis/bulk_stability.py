import json
import uuid

from pydash.objects import has, get

from pymatgen.core.structure import Structure

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)


@explicit_serialize
class BulkStabilityAnalysis(FiretaskBase):
    """
    Automated Stability Analysis Task to directly get,
    Thermodynamic and electrochemical stability of a given material.

    Args:


    Returns:
        Stability Analysis to DB
    """
    required_params = ["bulk_formula", "magnetic_ordering", "db_file"]
    optional_params = ["pbx_plot", "ehull_plot"]

    def run_task(self, fw_spec):

        # Variables
        bulk_formula = self["bulk_formula"]
        magnetic_ordering = self["magnetic_ordering"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        pbx_plot = self.get("pbx_plot", True)
        ehull_plot = self.get("ehull_plot", True)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Retrieve from DB
        docs = mmdb.collection.find_one({"task_label": "structure stactic"})
        
