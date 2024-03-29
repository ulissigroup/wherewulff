"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
class SurfacePourbaixDiagramAnalyzer(FiretaskBase):
    """
    Post-processing FireTask to plot the surface pourbaix diagram
    out of OH/Ox full coverage terminations

    Args:
        bulk_structure: pymatgen structure for getting bulk-formula
        db_file       : defult db.json file
        to_db         : option whether send the data to DB or not.

    Returns:
        plot for each surface and DB json data.
    """

    required_params = [
        "reduced_formula",
        "miller_index",
        "slab_uuid",
        "oriented_uuid",
        # "slab_hkl_uuid",
        "ads_slab_uuids",
        "db_file",
        "surface_pbx_uuid",
    ]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        self.reduced_formula = self["reduced_formula"]
        self.miller_index = self["miller_index"]
        slab_uuid = self["slab_uuid"]
        oriented_uuid = self["oriented_uuid"]
        # slab_hkl_uuid = self["slab_hkl_uuid"]
        ads_slab_uuids = self["ads_slab_uuids"]
        orig_ads_slab_uuids = self["ads_slab_uuids"]
        surface_pbx_uuid = self["surface_pbx_uuid"]

        # Get the dynamic adslab uuids from the fw_spec.
        # Note that this will be different from the orig_ads_slab_uuids
        # if the AdSlab Continuation is triggered from wall time issues
        ads_slab_uuids = [
            fw_spec[k]["adslab_uuid"]
            for k in fw_spec
            if f"{self.reduced_formula}-{self.miller_index}" in k
        ]

        # Create a new slab_hkl_uuid
        slab_hkl_uuid = str(slab_uuid) + "_" + str(self.miller_index)

        summary_dict = {
            "reduced_formula": self.reduced_formula,
            "miller_index": self.miller_index,
            "slab_uuid": slab_uuid,
            "oriented_uuid": oriented_uuid,
            "slab_hkl_uuid": slab_hkl_uuid,
            "orig_ads_slab_uuids": orig_ads_slab_uuids,
            "ads_slab_uuids": ads_slab_uuids,
        }

        # PBX Variables
        self.pH_range = list(range(0, 15, 1))
        kB = 0.0000861733  # eV/K
        Temp = 298.15  # Kelvin
        self.K = kB * Temp * np.log(10)  # Nernst slope
        self.reference_energies = {"H2O": -14.25994015, "H2": -6.77818501}

        # Surface PBX diagram uuid
        # surface_pbx_uuid = uuid.uuid4()
        summary_dict["surface_pbx_uuid"] = str(surface_pbx_uuid)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Find clean surface thru uuid
        doc_clean = mmdb.collection.find_one({"uuid": slab_uuid})

        slab_clean_obj_orig = Slab.from_dict(doc_clean["slab"])

        slab_clean = Structure.from_dict(
            doc_clean["calcs_reversed"][-1]["output"]["structure"]
        )

        slab_clean_energy = doc_clean["calcs_reversed"][-1]["output"]["energy"]
        slab_clean_comp = {
            str(key): value for key, value in slab_clean.composition.items()
        }

        # Find oriented_bulk thru uuid (since it is not decorated)
        # TODO: Mxide method should be compatible with decorated Elements (e.g. O2-)
        oriented_struct = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": str(oriented_uuid)})["output"][
                "structure"
            ]
        )

        oriented_wyckoffs = [
            site["properties"]["bulk_wyckoff"]
            for site in mmdb.db["tasks"].find_one({"uuid": str(slab_uuid)})["slab"][
                "oriented_unit_cell"
            ]["sites"]
        ]

        oriented_equivalents = [
            site["properties"]["bulk_equivalent"]
            for site in mmdb.db["tasks"].find_one({"uuid": str(slab_uuid)})["slab"][
                "oriented_unit_cell"
            ]["sites"]
        ]

        oriented_struct.add_site_property("bulk_wyckoff", oriented_wyckoffs)
        oriented_struct.add_site_property("bulk_equivalent", oriented_equivalents)

        # Filter by ads_slab_uuid and task_label
        ads_slab_terminations = {}
        dft_energy_oh_min = np.inf
        for n, ads_slab_uuid in enumerate(ads_slab_uuids):
            doc_ads = mmdb.collection.find_one({"uuid": ads_slab_uuid})
            ads_task_label = doc_ads["task_label"]
            adsorbate_label = ads_task_label.split("-")[2]  # OH_n or O_1
            if "OH_" in adsorbate_label:
                dft_energy_oh = doc_ads["calcs_reversed"][-1]["output"]["energy"]
                if dft_energy_oh <= dft_energy_oh_min:
                    dft_energy_oh_min = dft_energy_oh
                    # ads_task_label_oh_min = ads_task_label
                    ads_uuid_oh_min = ads_slab_uuid
                    n_oh_rotation = adsorbate_label.split("_")[1]

            if "O_" in adsorbate_label:
                dft_energy_ox = doc_ads["calcs_reversed"][-1]["output"]["energy"]
                # ads_task_label_ox = ads_task_label
                ads_uuid_ox = ads_slab_uuid

        ads_slab_terminations.update({str(ads_uuid_oh_min): dft_energy_oh_min})
        ads_slab_terminations.update({str(ads_uuid_ox): dft_energy_ox})

        summary_dict["ads_slab_terminations"] = ads_slab_terminations

        #### FW collection to retrieve site properties
        fw_collection = mmdb.db["fireworks"]
        fw_doc_oh = fw_collection.find_one({"spec.uuid": str(ads_uuid_oh_min)})
        # fw_doc_ox = fw_collection.find_one({"spec.uuid": str(ads_uuid_ox)})

        # Retrieve OH/Ox input struct and sort
        struct_oh_input = Structure.from_dict(
            fw_doc_oh["spec"]["_tasks"][0]["structure"]
        )
        struct_oh_input.sort()
        # struct_ox_input = Structure.from_dict(
        #    fw_doc_ox["spec"]["_tasks"][0]["structure"]
        # )

        # OH/Ox Structural info -> Compositions
        slab_oh = Structure.from_dict(
            mmdb.collection.find_one({"uuid": ads_uuid_oh_min})["calcs_reversed"][-1][
                "output"
            ]["structure"]
        )

        slab_clean, slab_clean_orig = self._add_site_properties(
            slab_clean, mmdb, uuid_termination=slab_uuid
        )

        slab_clean_obj = Slab(
            slab_clean.lattice,
            slab_clean.species,
            slab_clean.frac_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=oriented_struct,
            shift=slab_clean_obj_orig.shift,
            scale_factor=slab_clean_obj_orig.scale_factor,
            energy=slab_clean_energy,
            site_properties=slab_clean.site_properties,
        )

        # Appending site properties on slab_oh structure
        slab_oh, slab_oh_orig = self._add_site_properties(
            slab_oh, mmdb, uuid_termination=ads_uuid_oh_min
        )

        slab_oh_obj = Slab(
            slab_oh.lattice,
            slab_oh.species,
            slab_oh.frac_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=oriented_struct,
            shift=slab_clean_obj.shift,
            scale_factor=slab_clean_obj.scale_factor,
            energy=dft_energy_oh_min,
            site_properties=slab_oh.site_properties,
        )

        slab_oh_composition = {
            str(key): value for key, value in slab_oh.composition.items()
        }

        slab_ox = Structure.from_dict(
            mmdb.collection.find_one({"uuid": ads_uuid_ox})["calcs_reversed"][-1][
                "output"
            ]["structure"]
        )

        # Appending site properties on slab_ox structure
        slab_ox, slab_ox_orig = self._add_site_properties(
            slab_ox, mmdb, uuid_termination=ads_uuid_ox
        )

        slab_ox_obj = Slab(
            slab_ox.lattice,
            slab_ox.species,
            slab_ox.frac_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=oriented_struct,
            shift=slab_clean_obj.shift,
            scale_factor=slab_clean_obj.scale_factor,
            energy=dft_energy_ox,
            site_properties=slab_ox.site_properties,
        )

        slab_ox_composition = {
            str(key): value for key, value in slab_ox.composition.items()
        }

        # summary_dict["slab_clean"] = slab_clean_obj.as_dict()
        summary_dict[
            "slab_clean"
        ] = (
            slab_clean_obj.as_dict()
        )  # FIXME: You are not putting the relaxed slab since the structure is from the slab metadata
        summary_dict["slab_oh"] = slab_oh_obj.as_dict()
        summary_dict["slab_oh_orig"] = slab_oh_orig
        summary_dict["n_oh_rotation"] = n_oh_rotation
        summary_dict["slab_ox"] = slab_ox_obj.as_dict()
        summary_dict["slab_ox_orig"] = slab_ox_orig

        # Number of H2O and nH for PBX
        nH2O = slab_ox_composition["O"] - slab_clean_comp["O"]
        nH = slab_oh_composition["H"]
        nH_2 = 2.0 * nH

        # Graph Bounds - OER
        self.oer_std = self.oer_potential_std()
        self.oer_up = self.oer_potential_up()

        self.clean_2_OH = self._get_surface_potential_line(
            ads_slab_terminations[ads_uuid_oh_min], slab_clean_energy, nH=nH, nH2O=nH2O
        )

        # reference to the clean surface instead of OH-term
        self.OH_2_Ox = self._get_surface_potential_line(
            ads_slab_terminations[ads_uuid_ox],
            slab_clean_energy,
            nH=nH_2,
            nH2O=nH2O,
        )

        # Summary dict
        summary_dict["nH2O"] = nH2O
        summary_dict["nH"] = nH
        summary_dict["nH_2"] = nH_2
        summary_dict["oer_std"] = self.oer_std
        summary_dict["oer_up"] = self.oer_up
        summary_dict["clean_2_OH"] = self.clean_2_OH
        summary_dict["OH_2_Ox"] = self.OH_2_Ox

        # Plot the surface PBX diagram!
        pbx_plot = self._get_surface_pbx_diagram()
        pbx_plot.savefig(f"{self.reduced_formula}_{self.miller_index}_pbx.png", dpi=300)

        # Export json file
        with open(f"{self.reduced_formula}_{self.miller_index}_pbx.json", "w") as f:
            f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # To_DB
        if to_db:
            mmdb.collection = mmdb.db[
                f"{self.reduced_formula}-{self.miller_index}_surface_pbx"
            ]
            mmdb.collection.insert_one(summary_dict)

        # Logger
        logger.info(
            f"{self.reduced_formula}-({self.miller_index}) Surface Pourbaix Done!"
        )

        # Send the summary_dict to the child FW
        return FWAction(
            update_spec={
                f"{self.reduced_formula}_{self.miller_index}_surface_pbx": {
                    "slab_uuid": slab_uuid,
                    "reduced_formula": self.reduced_formula,
                    "miller_index": self.miller_index,
                    "slab_hkl_uuid": slab_hkl_uuid,
                    "oriented_uuid": oriented_uuid,
                    "ads_slabs_uuids": ads_slab_uuids,
                    "surface_pbx_uuid": surface_pbx_uuid,
                }
            },
            propagate=True,
        )

    def _add_site_properties(self, structure, mmdb, uuid_termination):
        """Abstracted way to add site properties into struct object"""
        # Copy structure object
        struct = structure.copy(site_properties=structure.site_properties)

        # Retrieve site properties depending on uuid
        slab_wyckoffs = [
            site["properties"]["bulk_wyckoff"]
            for site in mmdb.db["tasks"].find_one({"uuid": uuid_termination})["slab"][
                "sites"
            ]
        ]

        slab_equivalents = [
            site["properties"]["bulk_equivalent"]
            for site in mmdb.db["tasks"].find_one({"uuid": uuid_termination})["slab"][
                "sites"
            ]
        ]

        if (
            "surface_properties"
            in mmdb.db["tasks"].find_one({"uuid": uuid_termination})["slab"]["sites"][
                0
            ]["properties"]
        ):  # for clean term
            surface_properties = [
                site["properties"]["surface_properties"]
                for site in mmdb.db["tasks"].find_one({"uuid": uuid_termination})[
                    "slab"
                ]["sites"]
            ]
            struct.add_site_property("surface_properties", surface_properties)

        if (
            "binding_site"
            in mmdb.db["tasks"].find_one({"uuid": uuid_termination})["slab"]["sites"][
                0
            ]["properties"]
        ):  # for clean term
            binding_site = [
                site["properties"]["binding_site"]
                for site in mmdb.db["tasks"].find_one({"uuid": uuid_termination})[
                    "slab"
                ]["sites"]
            ]
            struct.add_site_property("binding_site", binding_site)

        if (
            "forces"
            in mmdb.db["tasks"].find_one({"uuid": uuid_termination})["slab"]["sites"][
                0
            ]["properties"]
        ):  # for clean term
            forces = [
                site["properties"]["forces"]
                for site in mmdb.db["tasks"].find_one({"uuid": uuid_termination})[
                    "slab"
                ]["sites"]
            ]
            struct.add_site_property("forces", forces)

        # Initialize from original magmoms
        # We need to use the uuid on the parent root node to get the input
        if (
            not len(
                mmdb.db["fireworks"].find_one({"spec.uuid": uuid_termination})["spec"][
                    "uuid_lineage"
                    ]
                ) 
                < 1
        ):
            orig_uuid = mmdb.db["fireworks"].find_one({"spec.uuid": uuid_termination})[
            "spec"
        ]["uuid_lineage"][0]

        else:
            orig_uuid = mmdb.db["fireworks"].find_one({"spec.uuid": uuid_termination})["spec"]["uuid"]

        orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_uuid})["orig_inputs"][
            "incar"
        ]["MAGMOM"]

        # Retrieve the original structure
        struct_orig = mmdb.db["tasks"].find_one({"uuid": orig_uuid})["input"][
            "structure"
        ]

        # Appending surface properties for slab object
        struct.add_site_property("bulk_wyckoff", slab_wyckoffs)
        struct.add_site_property("bulk_equivalent", slab_equivalents)
        struct.add_site_property("magmom", orig_magmoms)

        return struct, struct_orig

    def oer_potential_std(self):
        """
        Standard OER bound --> H2O -> O2 + 4H+ + 4e-
        """
        return [(1.229 - self.K * pH) for pH in self.pH_range]

    def oer_potential_up(self, u_0=1.60):
        """
        OER bound, U_0 selected by user and/or Experimental conditions.

        default: 1.60 V
        """
        return [(u_0 - self.K * pH) for pH in self.pH_range]

    def _get_surface_potential_line(
        self, dft_energy_specie, dft_energy_reference, nH=4, nH2O=4
    ):
        """
        Get line equation from:
            - clean --> OH:
                specie = OH-terminated
                reference = clean
            - OH --> Ox:
                specie = Ox-terminated
                reference = OH-terminated (and/or) clean
            - nH: Released (H+ + e-) PCETs
            - nH2O: How many water molecules are adsorbed on top of the clean (e.g. 110 = 4)
            - thermo_correction: Adding Thermo corrections depeding on the termination
        """
        intersept = (
            dft_energy_specie
            - dft_energy_reference
            - (nH2O * self.reference_energies["H2O"])
        ) * (1 / nH)
        intersept = intersept + (0.5 * self.reference_energies["H2"])
        return [(intersept - self.K * pH) for pH in self.pH_range]

    def _get_surface_pbx_diagram(self):
        """
        Matplotlib based function to build the PBX graph
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        # Axis labels
        ax.set_xlabel("pH", fontsize=12, fontweight="bold")
        ax.set_ylabel("U$_{SHE}$ (V)", fontsize=12, fontweight="bold")

        # Axis Limits
        ax.set_title(f"{self.reduced_formula}-({self.miller_index})")
        ax.set_xlim(0.0, 14.0)
        ax.set_ylim(0.0, 2.50)

        # OER bounds for standard and selected OER conditions
        ax.plot(self.pH_range, self.oer_std, linestyle="--", color="black")
        ax.plot(self.pH_range, self.oer_up, linestyle="--", color="red")

        # Surface PBX boundaries
        ax.plot(self.pH_range, self.clean_2_OH, linestyle="-", color="blue")
        ax.plot(self.pH_range, self.OH_2_Ox, linestyle="-", color="red")

        # Fill surface-terminations regions
        ax.fill_between(
            self.pH_range, self.clean_2_OH, color="blue", alpha=0.2, label="clean"
        )
        ax.fill_between(
            self.pH_range,
            self.clean_2_OH,
            self.OH_2_Ox,
            color="green",
            alpha=0.2,
            label="*OH",
        )
        ax.fill_between(
            self.pH_range, self.OH_2_Ox, 2.50, color="red", alpha=0.2, label="*Ox"
        )

        # Add legend
        plt.legend()

        return fig
