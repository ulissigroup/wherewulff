import uuid
import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.reactivity.oer import OER_SingleSite
from CatFlows.adsorption.adsorbate_configs import oer_adsorbates_dict
from CatFlows.workflows.oer_single_site import OERSingleSite_WF

# OER single site WF ?

@explicit_serialize
class OERSingleSiteFireTask(FiretaskBase):
    """
    OER Single Site FireTask.

    Args:

    Returns:
        OERSingleSite Firetaks.
    """

    required_params = ["reduced_formula", "miller_index", "vasp_cmd", "db_file"]
    optional_params = []

    def run_task(self, fw_spec):

        # Variables
        reduced_formula = self["reduced_formula"]
        miller_index = self["miller_index"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)

        # User-defined parameters !
        applied_potential = 1.60 # volts
        applied_pH = 0           # pH conditions
        user_point = np.array([applied_pH, applied_potential])

        parent_dict = fw_spec[f"{reduced_formula}_{miller_index}_surface_pbx"]
        surface_pbx_uuid = parent_dict["surface_pbx_uuid"]

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Get PBX collection from DB
        pbx_collection = mmdb.db[f"{reduced_formula}-{miller_index}_surface_pbx"]
        pbx_doc = pbx_collection.find_one({"surface_pbx_uuid": surface_pbx_uuid})

        # Decide most stable termination at given (V, pH)
        clean_2_oh_list = pbx_doc["clean_2_OH"] # clean -> OH 
        oh_2_ox_list = pbx_doc["OH_2_Ox"]       # OH -> Ox

        surface_termination = self._get_surface_stable_termination(user_point, 
                                                                   clean_2_oh_list, 
                                                                   oh_2_ox_list)

        # Retrieve the surface termination clean/OH/Ox geometries
        clean_surface = Slab.from_dict(pbx_doc["slab_clean"])
        stable_surface = Slab.from_dict(pbx_doc[f"slab_{surface_termination}"])


        # Generate OER single site intermediates (WNA)
        oer_wna = OER_SingleSite(stable_surface, adsorbates=oer_adsorbates_dict)
        oer_intermediates_dict = oer_wna.generate_oer_intermediates()

        # OER_WF
        oer_wf = OERSingleSite_WF(
                 oer_dict=oer_intermediates_dict,
                 slab=clean_surface,
                 slab_uuid=parent_dict["slab_uuid"],
                 oriented_uuid=parent_dict["oriented_uuid"],
                 vasp_cmd=vasp_cmd,
                 db_file=db_file
        )

        return FWAction(detours=[oer_wf])

    def _get_surface_stable_termination(self, user_point, clean_2_oh, oh_2_ox):
        """ 
        Helper function to detect whether a point lies above or below the pbx lines.
        READ This: https://math.stackexchange.com/a/274728
        """
        # Cross product
        is_above = lambda point,origin,end: np.cross(point-origin, end-origin) <= 0

        # Clean -> OH boundary
        clean_2_oh_origin = np.array([0, clean_2_oh[0]])
        clean_2_oh_end = np.array([14, clean_2_oh[-1]])

        above_clean = is_above(user_point, clean_2_oh_origin, clean_2_oh_end)

        # OH -> Ox boundary
        oh_2_ox_origin = np.array([0, oh_2_ox[0]])
        oh_2_ox_end = np.array([14, oh_2_ox[-1]])

        above_oh = is_above(user_point, oh_2_ox_origin, oh_2_ox_end)

        # decide
        if above_clean == False:
            surface_termination = "clean"
        elif above_clean == True and above_oh == False:
            surface_termination = "oh"
        elif above_clean == True and above_oh == True:
            surface_termination = "ox"
    
        return surface_termination

