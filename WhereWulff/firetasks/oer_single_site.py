import uuid
import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE

from WhereWulff.reactivity.oer import OER_SingleSite
from WhereWulff.adsorption.adsorbate_configs import oer_adsorbates_dict
from WhereWulff.workflows.oer_single_site import OERSingleSite_WF
from atomate.utils.utils import env_chk

logger = get_logger(__name__)


@explicit_serialize
class OERSingleSiteFireTask(FiretaskBase):
    """
    OER Single Site FireTask.

    Args:
        reduced_formula (str): Reduced formula of the given material e.g RuO2
        miller_index    (str): Miller index of the given surface e.g 110
        metal_site      (str): OER site composition (selecting the metal)
        vasp_cmd        (env): Environment variable to call vasp
        db_file         (env): Environment variable to connect to the DB

    Returns:
        OERSingleSite Firetaks.
    """

    required_params = [
        "reduced_formula",
        "miller_index",
        "slab_orig",
        "bulk_like_sites",
        "ads_dict_orig",
        "metal_site",
        "applied_potential",
        "applied_pH",
        "vasp_cmd",
        "db_file",
        "run_fake",
        "surface_pbx_uuid",
        # "streamline",
        "checkpoint_path",
    ]
    optional_params = []

    def run_task(self, fw_spec):

        # Variables
        reduced_formula = self["reduced_formula"]
        miller_index = self["miller_index"]
        # slab_orig = self["slab_orig"]
        bulk_like_sites = self["bulk_like_sites"]
        ads_dict_orig = self["ads_dict_orig"]
        metal_site = self["metal_site"]
        applied_potential = self["applied_potential"]
        checkpoint_path = env_chk(self["checkpoint_path"], fw_spec)  # abstract variable
        applied_pH = self["applied_pH"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        run_fake = self.get("run_fake", False)
        surface_pbx_uuid = self["surface_pbx_uuid"]
        # streamline = self.get("streamline", False)

        # User-defined parameters !
        # applied_potential = 1.60  # volts
        # applied_pH = 0  # pH conditions
        user_point = np.array([applied_pH, applied_potential])
        # ORR
        #        user_point[1] = 0.9

        parent_dict = fw_spec[f"{reduced_formula}_{miller_index}_surface_pbx"]
        surface_pbx_uuid = parent_dict["surface_pbx_uuid"]

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Get PBX collection from DB
        pbx_collection = mmdb.db[f"{reduced_formula}-{miller_index}_surface_pbx"]
        pbx_doc = pbx_collection.find_one({"surface_pbx_uuid": surface_pbx_uuid})

        # Decide most stable termination at given (V, pH)
        clean_2_oh_list = pbx_doc["clean_2_OH"]  # clean -> OH
        oh_2_ox_list = pbx_doc["OH_2_Ox"]  # OH -> Ox

        surface_termination = self._get_surface_stable_termination(
            user_point, clean_2_oh_list, oh_2_ox_list
        )

        # breakpoint()
        # Retrieve the surface termination clean/OH/Ox geometries
        clean_surface = Slab.from_dict(pbx_doc["slab_clean"])
        stable_surface = Slab.from_dict(pbx_doc[f"slab_{surface_termination}"])

        # Retrieve the surface termination as input
        if surface_termination == "ox":
            stable_surface_orig = ads_dict_orig["O_1"]
        elif surface_termination == "oh":
            n_oh_rotation = pbx_doc["n_oh_rotation"]
            stable_surface_orig = ads_dict_orig[f"OH_{n_oh_rotation}"]
        else:
            stable_surface_orig = self["slab_orig"]  # Original pristine surface

        # Generate OER single site intermediates (WNA)
        oer_wna = OER_SingleSite(
            stable_surface,
            slab_orig=stable_surface_orig,
            slab_clean=clean_surface,
            bulk_like_sites=bulk_like_sites,
            metal_site=metal_site,
            adsorbates=oer_adsorbates_dict,
            # streamline=streamline,
            surface_coverage=surface_termination,
            checkpoint_path=checkpoint_path,
        )
        oer_intermediates_dict = oer_wna.generate_oer_intermediates()

        # Logger
        logger.info(
            f"{reduced_formula}-{miller_index} at (pH = {applied_pH}, V = {applied_potential} is: {surface_termination}"
        )
        oer_wfs = []
        # OER_WF # We need a workflow for each ref_slab
        for site in oer_wna.ads_indices:
            oer_intermediates = {
                k: v for k, v in oer_intermediates_dict.items() if str(site) in k
            }  # Segment by site
            oer_wf = OERSingleSite_WF(
                oer_dict=oer_intermediates,
                slab=clean_surface,
                metal_site=metal_site,
                slab_uuid=parent_dict["slab_uuid"],
                oriented_uuid=parent_dict["oriented_uuid"],
                surface_termination=surface_termination,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
                run_fake=run_fake,
                surface_pbx_uuid=surface_pbx_uuid,
            )
            oer_wf.name = oer_wf.name + f"_{site}"
            for fw in oer_wf.fws:
                if "Analysis" in fw.name:
                    fw.name = fw.name + f"_{site}"
            oer_wfs.append(oer_wf)
        return FWAction(detours=oer_wfs)

    def _get_surface_stable_termination(self, user_point, clean_2_oh, oh_2_ox):
        """
        Helper function to detect whether a point lies above or below the pbx lines.
        READ This: https://math.stackexchange.com/a/274728
        """
        # Cross product
        is_above = (
            lambda point, origin, end: np.cross(point - origin, end - origin) <= 0
        )

        # Clean -> OH boundary
        clean_2_oh_origin = np.array([0, clean_2_oh[0]])
        clean_2_oh_end = np.array([14, clean_2_oh[-1]])

        above_clean = is_above(user_point, clean_2_oh_origin, clean_2_oh_end)

        # OH -> Ox boundary
        oh_2_ox_origin = np.array([0, oh_2_ox[0]])
        oh_2_ox_end = np.array([14, oh_2_ox[-1]])

        above_oh = is_above(user_point, oh_2_ox_origin, oh_2_ox_end)

        # decide
        if not above_clean:
            surface_termination = "clean"
        elif above_clean and not above_oh:
            surface_termination = "oh"
        elif above_clean and above_oh:
            surface_termination = "ox"

        return surface_termination
