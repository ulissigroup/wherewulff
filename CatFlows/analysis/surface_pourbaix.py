import json
import uuid

from pydash.objects import has, get

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)

@explicit_serialize
class SurfacePourbaixDiagram(FiretaskBase):
    """
    Post-processing FireTask to plot the surface pourbaix diagram
    out of OH/Ox full coverage terminations

    Args:

    Returns:
        plot for each surface and DB json data.
    """

    requiered_params = ["bulk_structure", "db_file"]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        bulk_structure = self["bulk_structure"]

        summary_dict = {}

        # Surface PBX diagram uuid
        surface_pbx_uuid = uuid.uuid4()
        summary_dict["surface_pbx_uuid"] = surface_pbx_uuid

        # Bulk formula
        bulk_formula = bulk_structure.composition.reduced_formula

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Find clean surfaces 
        collection_clean = mmdb.db["surface_energies"]
        docs_clean = collection_clean.find({"task_label": {"$regex": f"{bulk_formula}_.*_surface_energy"}})

        # Find OH/Ox terminations per surface
        collection_term = mmdb.db["tasks"]
        docs_OH = collection_term.find({"task_label": {"$regex": f"{bulk_formula}-.*-OH_.*"}})
        docs_Ox = collection_term.find({"task_label": {"$regex": f"{bulk_formula}-.*-O_.*"}})

        # Get miller indices and DFT Energies
        surface_clean_dict = {}
        for d_clean in docs_clean:
            miller_index = d_clean["miller_index"]
            dft_energy_clean = d_clean["slab_E"]
            surface_clean_dict.update({str(miller_index): dft_energy_clean})

        # List of (hkl)
        miller_indices = list(surface_clean_dict.keys())

        # Filter OH termination by DFT energy
        # nested dict as {"hkl": {task_label: energy}}
        task_label_oh_dict = {}
        for hkl in miller_indices:
            task_label_oh_dict[hkl] = {}
            for d_OH in docs_OH:
                task_label = d_OH["task_label"]
                surf_orientation = task_label.split("-")[1]
                dft_energy_oh = d_OH["calcs_reversed"][-1]["output"]["energy"]
                task_label_oh_dict[surf_orientation].update({str(task_label): dft_energy_oh})

        # 1st key per hkl should be the lowest one
        task_label_oh_sort = {key: sorted(values, key=lambda x:x[1]) for key, values in task_label_oh_dict.items()}

        oh_task_labels = [list(v[k])[0] for k, v in task_label_oh_sort.items()]




        












        


