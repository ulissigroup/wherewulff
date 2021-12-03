import json
import uuid

import numpy as np

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import Slab

from pydash.objects import has, get

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)


@explicit_serialize
class OER_SingleSiteAnalyzer(FiretaskBase):
    """
    Post-processing FireTask to derive Delta G's for
    OER (WNA) mechanism in a single active site and
    derive the theoretical overpotential.

    Args:
        db_file (env): Environment variable to connect to the DB.


    Returns:
        Reactivity post-processing for a given surface
        and DB json data.
    """

    required_params = [
        "reduced_formula",
        "miller_index",
        "slab_uuid",
        "ads_slab_uuids",
        #"surface_termination", - might be useful to filter out
        "db_file",
    ]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        self.reduced_formula = self["reduced_formula"]
        self.miller_index = self["miller_index"]
        slab_uuid = self["slab_uuid"]
        ads_slab_uuids = self["ads_slab_uuids"]

        # Get the dynamic adslab uuids from the fw_spec.
        # Note that this will be different from the orig_ads_slab_uuids
        # if the AdSlab Continuation is triggered from wall time issues
        ads_slab_uuids = [
            fw_spec[k]["adslab_uuid"]
            for k in fw_spec
            if f"{self.reduced_formula}-{self.miller_index}" in k
        ]

        # Summary dict
        summary_dict = {
            "reduced_formula": self.reduced_formula,
            "miller_index": self.miller_index,
            "slab_uuid": slab_uuid,
            "ads_slab_uuids": ads_slab_uuids,
        }

        # OER variables
        self.ref_energies = {"H2O": -14.25994015, "H2": -6.77818501}

        # Reactivity uuid
        oer_single_site_uuid = uuid.uuid4()
        summary_dict["oer_single_site"] = str(oer_single_site_uuid)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Filter min energy per intermediate
        oer_intermediates = {}
        for n, ads_slab_uuid in enumerate(ads_slab_uuids):
            doc_oer = mmdb.collection.find_one({"uuid": ads_slab_uuid})
            oer_task_label = doc_oer["task_label"]
            adsorbate_label = oer_task_label.split("-")[2] # reference, OH_n, O_n, OOH_up_n, OOH_down_n
            if "reference" in adsorbate_label:
                dft_energy_reference = doc_oer["calcs_reversed"][-1]["output"]["energy"]
                oer_uuid_reference = ads_slab_uuid
                oer_intermediates["reference"] = dft_energy_reference





        # Export to json file
        with open(f"{self.reduced_formula}_{self.miller_index}_oer.json", "w") as f:
            f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # To DB -> (This should be unique every time)
        if to_db:
            mmdb.collection = mmdb.db[
                f"{self.reduced_formula}-{self.miller_index}_oer_single_site"
            ]
            mmdb.collection.insert_one(summary_dict)

        # Logger
        logger.info(
            f"{self.reduced_formula}-{self.miller_index} -> (overpotential: {overpotential}, PDS: {pot_det_step})"
        )

        # Send the summary_dict to the child FW (?)
        return FWAction(
            update_spec={},
            propagate=True,
        )

    def Eads_OH(self, energy_oh, energy_clean, thermo_correction=None):
        """
        Reaction H2O + (*) --> OH* + H+ + e-
        Args:
            energy_oh
            energy_clean
            thermo_correction
        Returns:
            Delta G(OH) value
        """
        eads_oh = (
            energy_oh
            - energy_clean
            - (self.ref_energies["H2O"] - (0.5 * self.ref_energies["H2"]))
        )
        if thermo_correction:
            eads_oh = eads_oh + thermo_correction
            return eads_oh
        else:
            return eads_oh

    def Eads_Ox(self, energy_ox, energy_clean, thermo_correction=None):
        """
        Reaction: OH* --> O* + H+ + e-
        Args:
            energy_ox
            energy_clean
            thermo_correction
        Returns:
            Delta G(Ox) value
        """
        eads_ox = (
            energy_ox
            - energy_clean
            - (self.ref_energies["H2O"] - self.ref_energies["H2"])
        )
        if thermo_correction:
            eads_ox = eads_ox + thermo_correction
            return eads_ox
        else:
            return eads_ox

    def Eads_OOH(self, energy_ooh, energy_clean, thermo_correction=None):
        """
        Reaction: O* + H2O --> OOH* + H+ + e-
        Args:
            energy_ooh
            energy_clean
            thermo_correction
        Returns:
            Delta G(OOH) value
        """
        eads_ooh = (
            energy_ooh
            - energy_clean
            - ((2 * self.ref_energies["H2O"]) - (1.5 * self.ref_energies["H2"]))
        )
        if thermo_correction:
            eads_ooh = eads_ooh + thermo_correction
            return eads_ooh
        else:
            return eads_ooh

    def oxygen_evolution(self, eads_ooh, std_potential=4.92):
        """
        Reaction: OOH* --> O2(g) + (*) + H+ + e-
        Args:
            eads_ooh
            std_potential (default: 4.92)
        Returns:
            Delta G of the O2 evolution (last step)
        """
        o2_release = std_potential - eads_ooh
        return o2_release

    def linear_relationships_and_overpotential(self, delta_g_dict):
        """
        Computes linear relationships and derives theoretical overpotential
        Args:
            delta_g_dict = {"g_oh", "g_ox", "g_ooh", "g_o2"}
        Returns:
            Dictionary with linear relationships and overpotential.
        """
        # Linear relationships
        ox_oh = delta_g_dict["g_ox"] - delta_g_dict["g_oh"]
        ooh_ox = delta_g_dict["g_ooh"] - delta_g_dict["g_ox"]
        linear_relationships_dict = {
            "g_oh": delta_g_dict["g_oh"],
            "ox_oh": ox_oh,
            "ooh_ox": ooh_ox,
            "g_o2": delta_g_dict["g_o2"],
        }

        # Find max in linear_rel. dict
        find_max_step = max(
            linear_relationships_dict, key=linear_relationships_dict.get
        )

        # Theoretical overpotential
        oer_overpotential = linear_relationships_dict[find_max_step] - 1.23

        # Result Dict
        result_dict = {**delta_g_dict, **linear_relationships_dict}
        result_dict["overpotential"] = oer_overpotential
        result_dict["PDS"] = find_max_step

        return result_dict
