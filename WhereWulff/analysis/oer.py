import json
import uuid

import numpy as np
import re

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
        reduced_formula     (e.g IrO2)   : Formula of the given material.
        miller_index        (e.g [1,1,0]): Crystallographic orientation from slab model.
        metal_site          (e.g Ir)     : Targeted element as reactive site in the slab model.
        slab_uuid           (str)        : Unique hash to identify previous jobs in the same run.
        ads_slabs_uuid      (str)        : Unique hashes from PBX to identify those DFT runs.
        surface_termination (str)        : Either clean, OH or Ox from Surface Pourbaix workflow.
        db_file             (env)        : Environment variable to connect to the DB.

    Returns:
        OER Single site Reactivity post-processing for a given surface
        and DB json data.
    """

    required_params = [
        "reduced_formula",
        "miller_index",
        "metal_site",
        "slab_uuid",
        "ads_slab_uuids",
        "surface_termination",
        "db_file",
        "surface_pbx_uuid",
    ]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):
        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        db_file = "/Users/yurisanspeur/atomate/config/db.json"
        to_db = self.get("to_db", True)
        self.reduced_formula = self["reduced_formula"]
        self.miller_index = self["miller_index"]
        self.metal_site = self["metal_site"]
        slab_uuid = self["slab_uuid"]
        ads_slab_uuids = self["ads_slab_uuids"]
        surface_termination = self["surface_termination"]
        surface_pbx_uuid = self["surface_pbx_uuid"]

        # parent_dict = fw_spec[f"{self.reduced_formula}_{self.miller_index}_surface_pbx"]
        # surface_pbx_uuid = parent_dict["surface_pbx_uuid"]

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
            "metal_site": self.metal_site,
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

        # Retrieve the surface termination from pbx collection
        pbx_collection = mmdb.db[
            f"{self.reduced_formula}-{self.miller_index}_surface_pbx"
        ]
        doc_termination = pbx_collection.find_one(
            {"surface_pbx_uuid": surface_pbx_uuid}
        )

        stable_termination = Slab.from_dict(
            doc_termination[f"slab_{surface_termination}"]
        )  # ox or oh

        # Filter OER intermediates (reference, OH_n, O_n, OOH_up_n, OOH_down_n)
        oer_intermediates_uuid, oer_intermediates_energy = {}, {}
        (
            dft_energy_oh_min,
            dft_energy_ooh_up_min,
            dft_energy_ooh_down_min,
            dft_energy_ooh_min,
        ) = (
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        )
        for n, ads_slab_uuid in enumerate(ads_slab_uuids):
            doc_oer = mmdb.collection.find_one({"uuid": ads_slab_uuid})
            oer_task_label = doc_oer["task_label"]
            adsorbate_label = oer_task_label.split("-")[3]
            # oer_task_label = mmdb.db["fireworks"].find_one(
            #    {"spec.uuid": doc_oer["uuid"]}
            # )["name"]
            # adsorbate_label = oer_task_label.split("-")[4]
            # reference active site
            if "reference" in adsorbate_label:
                dft_energy_reference = doc_oer["calcs_reversed"][0]["output"][
                    "ionic_steps"
                ][-1]["e_0_energy"]
                oer_uuid_reference = ads_slab_uuid
                oer_intermediates_uuid["reference"] = oer_uuid_reference
                oer_intermediates_energy["reference"] = dft_energy_reference
            # select OH intermediate as min dft energy
            if re.match("^OH_.*", adsorbate_label):
                dft_energy_oh = doc_oer["calcs_reversed"][0]["output"]["ionic_steps"][
                    -1
                ]["e_0_energy"]
                if dft_energy_oh <= dft_energy_oh_min:
                    dft_energy_oh_min = dft_energy_oh
                    oer_uuid_oh = adsorbate_label
                    oer_intermediates_uuid["OH"] = oer_uuid_oh
                    oer_intermediates_energy["OH"] = dft_energy_oh_min
            # select Ox intermediate as min dft energy (no rotation - just one)
            if "O_" in adsorbate_label:
                dft_energy_oh = doc_oer["calcs_reversed"][0]["output"]["ionic_steps"][
                    -1
                ]["e_0_energy"]
                oer_uuid_ox = adsorbate_label
                oer_intermediates_uuid["Ox"] = oer_uuid_ox
                oer_intermediates_energy["Ox"] = dft_energy_oh
            # select OOH_up intermediate as min dft energy
            if re.match("^OOH_up_.*", adsorbate_label):
                dft_energy_ooh_up = doc_oer["calcs_reversed"][0]["output"][
                    "ionic_steps"
                ][-1]["e_0_energy"]
                if dft_energy_ooh_up <= dft_energy_ooh_up_min:
                    oer_uuid_ooh_up = adsorbate_label
                    dft_energy_ooh_up_min = dft_energy_ooh_up
                    oer_intermediates_uuid["OOH_up"] = oer_uuid_ooh_up
                    oer_intermediates_energy["OOH_up"] = dft_energy_ooh_up_min
            # select OOH_down intermediate as min dft energy
            if "OOH_down_" in adsorbate_label:
                dft_energy_ooh_down = doc_oer["calcs_reversed"][0]["output"][
                    "ionic_steps"
                ][-1]["e_0_energy"]
                if dft_energy_ooh_down <= dft_energy_ooh_down_min:
                    oer_uuid_ooh_down = adsorbate_label
                    dft_energy_ooh_down_min = dft_energy_ooh_down
                    oer_intermediates_uuid["OOH_down"] = oer_uuid_ooh_down
                    oer_intermediates_energy["OOH_down"] = dft_energy_ooh_down_min
            if re.match("^OOH_.*", adsorbate_label):  # streamline case
                dft_energy_ooh = doc_oer["calcs_reversed"][0]["output"]["ionic_steps"][
                    -1
                ]["e_0_energy"]
                if dft_energy_ooh <= dft_energy_ooh_min:
                    oer_uuid_ooh = adsorbate_label
                    dft_energy_ooh_min = dft_energy_ooh
                    oer_intermediates_uuid["OOH"] = oer_uuid_ooh
                    oer_intermediates_energy["OOH"] = dft_energy_ooh
            if re.match(".*o2_star.*", adsorbate_label):
                dft_energy_o2_star = doc_oer["calcs_reversed"][0]["output"][
                    "ionic_steps"
                ][-1]["e_0_energy"]
                oer_intermediates_uuid["o2_star"] = adsorbate_label
                oer_intermediates_energy["o2_star"] = dft_energy_o2_star

        # Add termination as OER intermediate
        if surface_termination == "ox":
            oer_intermediates_energy["Ox"] = stable_termination.energy

        if surface_termination == "oh":
            oer_intermediates_energy["OH"] = stable_termination.energy

        # Add both oer dicts into summary_dict
        summary_dict["oer_uuids"] = oer_intermediates_uuid
        summary_dict["oer_energies"] = oer_intermediates_energy

        # Compute delta G
        delta_g_oer_dict = {}

        # Eads_OH
        eads_oh = self.Eads_OH(
            oer_intermediates_energy["OH"],
            oer_intermediates_energy["reference"],
            thermo_correction=0.295,
        )

        delta_g_oer_dict["g_oh"] = eads_oh

        # Eads_Ox
        eads_ox = self.Eads_Ox(
            oer_intermediates_energy["Ox"],
            oer_intermediates_energy["reference"],
            thermo_correction=0.044,
        )

        # Eads_OOH
        delta_g_oer_dict["g_ox"] = eads_ox
        if (
            "OOH_up" in oer_intermediates_energy
            and "OOH_down" in oer_intermediates_energy
        ):
            eads_ooh_up = self.Eads_OOH(
                oer_intermediates_energy["OOH_up"],
                oer_intermediates_energy["reference"],
                thermo_correction=0.377,
            )

            eads_ooh_down = self.Eads_OOH(
                oer_intermediates_energy["OOH_down"],
                oer_intermediates_energy["reference"],
                thermo_correction=0.377,
            )
            # Select between OOH_up and OOH_down
            if eads_ooh_up <= eads_ooh_down:
                delta_g_oer_dict["g_ooh"] = eads_ooh_up

            if eads_ooh_down <= eads_ooh_up:
                delta_g_oer_dict["g_ooh"] = eads_ooh_down
        else:
            eads_ooh = self.Eads_OOH(
                oer_intermediates_energy["OOH"],
                oer_intermediates_energy["reference"],
                thermo_correction=0.377,
            )
            delta_g_oer_dict["g_ooh"] = eads_ooh

        # O2 evolution
        o2_evol = self.oxygen_evolution(delta_g_oer_dict["g_ooh"], std_potential=4.92)
        # Eads_o2_star
        eads_o2_star = self.Eads_o2_star(
            oer_intermediates_energy["o2_star"], oer_intermediates_energy["reference"]
        )
        delta_g_oer_dict["g_o2_star"] = eads_o2_star

        delta_g_oer_dict["g_o2"] = o2_evol

        # Linear Relationships - Theoretical overpotential - PDS
        oer_dict = self.linear_relationships_and_overpotential(delta_g_oer_dict)
        overpotential = oer_dict["overpotential"]
        pds_step = oer_dict["PDS"]
        # Add oer_dict to summary_dict
        summary_dict["oer_info"] = oer_dict
        summary_dict["overpotential"] = overpotential
        summary_dict["PDS"] = pds_step
        # Export to json file
        with open(
            f"{self.reduced_formula}_{self.miller_index}_{self.metal_site}_oer.json",
            "w",
        ) as f:
            f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # To DB -> (This should be unique every time)
        if to_db:
            mmdb.collection = mmdb.db[
                f"{self.reduced_formula}-{self.miller_index}_{self.metal_site}_oer_single_site"
            ]
            mmdb.collection.insert_one(summary_dict)

        # Logger
        logger.info(
            f"{self.reduced_formula}-{self.miller_index} -> (overpotential: {overpotential}, PDS: {pds_step})"
        )

        def func(U_SHE, C, U_0, E_0):
            Ghat = -0.5 * C * (U_SHE - U_0) ** 2 + E_0
            return Ghat

        Gs_OH_fxn_U = None
        Gs_Ox_fxn_U = None
        # EDL calcs and plots
        for edl_uuid in ads_slab_uuids:
            fw = mmdb.db["fireworks"].find_one(
                {
                    "$and": [
                        {"spec._tasks.0.replace_uuid": edl_uuid},
                        {"name": {"$regex": ".*EDL_analysis.*"}},
                    ]
                }
            )
            launch_id = fw["launches"][0]
            name = mmdb.db["fireworks"].find_one({"spec.uuid": edl_uuid})["name"]
            # regex
            edl_results = mmdb.db["launches"].find_one({"launch_id": launch_id})[
                "action"
            ]["stored_data"]
            Us = np.linspace(0, 1, 100)
            fit_params = edl_results["fit_params"]
            if re.match(".*-O_.*", name):
                Gs_Ox_fxn_U = np.array(func(Us, *fit_params))
            elif re.match(".*-reference.*", name):
                Gs_reference_fxn_U = np.array(func(Us, *fit_params))
            elif re.match(".*-OOH.*", name):
                Gs_OOH_fxn_U = np.array(func(Us, *fit_params))
            elif re.match(".*-OH.*", name):
                Gs_OH_fxn_U = np.array(func(Us, *fit_params))
            elif re.match(".*o2_star.*", name):
                Gs_o2_star_fxn_U = np.array(func(Us, *fit_params))

        if (
            Gs_OH_fxn_U is None
        ):  # means termination was *OH and have to retrieve from previous calc
            edl_uuid_oh = min(
                doc_termination["ads_slab_terminations"],
                key=doc_termination["ads_slab_terminations"].get,
            )
            fw = mmdb.db["fireworks"].find_one(
                {
                    "$and": [
                        {"spec._tasks.0.replace_uuid": edl_uuid_oh},
                        {"name": {"$regex": ".*EDL_analysis.*"}},
                    ]
                }
            )
            launch_id = fw["launches"][0]
            name = mmdb.db["fireworks"].find_one({"spec.uuid": edl_uuid_oh})["name"]
            # regex
            edl_results = mmdb.db["launches"].find_one({"launch_id": launch_id})[
                "action"
            ]["stored_data"]
            Us = np.linspace(0, 1, 100)
            fit_params = edl_results["fit_params"]
            Gs_OH_fxn_U = np.array(func(Us, *fit_params))
        elif Gs_Ox_fxn_U is None:
            edl_uuid_ox = max(
                doc_termination["ads_slab_terminations"],
                key=doc_termination["ads_slab_terminations"].get,
            )
            fw = mmdb.db["fireworks"].find_one(
                {
                    "$and": [
                        {"spec._tasks.0.replace_uuid": edl_uuid_ox},
                        {"name": {"$regex": ".*EDL_analysis.*"}},
                    ]
                }
            )
            launch_id = fw["launches"][0]
            name = mmdb.db["fireworks"].find_one({"spec.uuid": edl_uuid_ox})["name"]
            # regex
            edl_results = mmdb.db["launches"].find_one({"launch_id": launch_id})[
                "action"
            ]["stored_data"]
            Us = np.linspace(0, 1, 100)
            fit_params = edl_results["fit_params"]
            Gs_Ox_fxn_U = np.array(func(Us, *fit_params))

        # Compute binding energies as fxn of U
        Eads_Ox_fxn_U = (
            Gs_Ox_fxn_U
            - Gs_reference_fxn_U
            - (self.ref_energies["H2O"] - self.ref_energies["H2"])
        ) + 0.044

        Eads_OOH_fxn_U = (
            Gs_OOH_fxn_U
            - Gs_reference_fxn_U
            - ((2 * self.ref_energies["H2O"]) - (1.5 * self.ref_energies["H2"]))
            + 0.377
        )
        Eads_OH_fxn_U = (
            Gs_OH_fxn_U
            - Gs_reference_fxn_U
            - (self.ref_energies["H2O"] - (0.5 * self.ref_energies["H2"]))
            + 0.295
        )
        Eads_o2_star_fxn_U = (
            Gs_o2_star_fxn_U
            - Gs_reference_fxn_U
            - (2 * self.ref_energies["H2O"] - 2 * self.ref_energies["H2"])
        )
        Gs_Ox_OH = Eads_Ox_fxn_U - Eads_OH_fxn_U
        Gs_OOH_Ox = Eads_OOH_fxn_U - Eads_Ox_fxn_U
        Gs_o2_star_OOH = Eads_o2_star_fxn_U - Eads_OOH_fxn_U
        Gs_star_o2_star = 4.92 - Eads_o2_star_fxn_U
        GO2 = 4.92 - Eads_OOH_fxn_U

        stacked_Gs = np.vstack((Eads_OH_fxn_U, Gs_Ox_OH, Gs_OOH_Ox, Gs_o2_star_OOH, Gs_star_o2_star))
        eta_fxn_U = 1.23 - stacked_Gs.min(axis=0)
        breakpoint()
        from matplotlib import pyplot as plt

        fig_eta = plt.figure(figsize=(10.0, 10.0))
        ax_eta = fig_eta.add_subplot(111)
        ax_eta.plot(Us, eta_fxn_U, "b-")
        ax_eta.set_xlabel("U SHE (V)")
        ax_eta.set_ylabel("Overpotential (V RHE)")
        fig_eta.savefig("eta_fxn_U.png")
        fig_pds = plt.figure(figsize=(10.0, 10.0))
        ax_pds = fig_pds.add_subplot(111)
        ax_pds.plot(Us, GO2, "r-", label="GO2")
        ax_pds.plot(Us, Gs_OOH_Ox, "b-", label="GOOH-GOx")
        ax_pds.plot(Us, Eads_OH_fxn_U, "m-", label="GOH")
        ax_pds.plot(Us, Gs_Ox_OH, "g", label="GOx-GOH")
        ax_pds.set_xlabel("U SHE (V)")
        ax_pds.set_ylabel("Elementary Reaction Free Energy (eV)")
        ax_pds.legend(loc="best")
        fig_pds.savefig("PDS_fxn_U.png")

        # Send the summary_dict to the child FW (?)
        return FWAction(
            stored_data={
                f"{self.reduced_formula}_{self.miller_index}_{self.metal_site}_oer_single_site": {
                    "oer_single_site_uuid": str(oer_single_site_uuid),
                    "overpotential": overpotential,
                    "PDS": pds_step,
                    "oer_info": oer_dict,
                }
            }
        )

    # TODO: Abstract the min energy
    def _get_min_energy_intermediate(self):
        """Returns min DFT energy across same intermediate"""
        return

    def Eads_o2_star(self, energy_o2_star, energy_ref, thermo_correction=None):
        eads_o2_star = (
            energy_o2_star
            - energy_ref
            - (2 * self.ref_energies["H2O"] - 2 * self.ref_energies["H2"])
        )
        if thermo_correction:
            eads_o2_star = eads_o2_star + thermo_correction
            return eads_o2_star
        else:
            return eads_o2_star

    def Eads_OH(self, energy_oh, energy_clean, thermo_correction=None):
        """
        Reaction: H2O + (*) --> OH* + H+ + e-
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
        o2_star_ooh = delta_g_dict["g_o2_star"] - delta_g_dict["g_ooh"]
        star_o2_star = 4.92 - delta_g_dict["g_o2_star"]

        linear_relationships_dict = {
            "g_oh": delta_g_dict["g_oh"],
            "ox_oh": ox_oh,
            "ooh_ox": ooh_ox,
            "o2_star_ooh": o2_star_ooh,
            "*_o2_star": star_o2_star,
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
