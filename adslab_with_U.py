import uuid
import numpy as np
from CatFlows.dft_settings.settings import MOSurfaceSet
from pydash.objects import has, get
from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator
from CatFlows.workflows.surface_pourbaix import (
    add_adsorbates,
    get_angles,
    get_clockwise_rotations,
    _bulk_like_adsites_perturbation,
)

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE
from CatFlows.fireworks.optimize import AdsSlab_FW


@explicit_serialize
class OptimizeAdslabsWithU(FiretaskBase):
    """We retrieve the relaxed slab at the +U and use that as a
    starting point for the adslab relaxations, with rotational sweep at
    that same value of U"""

    required_params = [
        "reduced_formula",
        "adsorbates",
        "db_file",
        "miller_index",
        "U_values",
        "vis",
    ]
    optional_params = ["adsite_index"]

    def run_task(self, fw_spec):
        vis = self["vis"]
        adsite_index = self["adsite_index"]
        adsorbates = self["adsorbates"]
        db_file = env_chk(self["db_file"], fw_spec)
        U_values = self[
            "U_values"
        ]  # FIXME: I think this needs to be a dictionary of element with values of +U
        reduced_formula = self["reduced_formula"]
        miller_index = self["miller_index"]
        #        bulk_slab_key = "_".join([reduced_formula, miller_index])
        # Get the slab uuid
        slab_uuid = fw_spec.get("slab_uuid")
        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        slab_wyckoffs = [
            site["properties"]["bulk_wyckoff"]
            for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"]["sites"]
        ]
        slab_equivalents = [
            site["properties"]["bulk_equivalent"]
            for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"]["sites"]
        ]
        slab_forces = mmdb.db["tasks"].find_one({"uuid": slab_uuid})["output"]["forces"]
        slab_struct = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": slab_uuid})["output"]["structure"]
        )
        # Retrieve original structure from the root node via the uuid_lineage field
        # in the spec of the terminal node
        if (
            not len(
                mmdb.db["fireworks"].find_one({"spec.uuid": slab_uuid})["spec"][
                    "uuid_lineage"
                ]
            )
            < 1
        ):
            orig_slab_uuid = mmdb.db["fireworks"].find_one({"spec.uuid": slab_uuid})[
                "spec"
            ]["uuid_lineage"][
                0
            ]  # FIXME: Think it is better to get the orig input from the WriteIOSet task itself
        else:
            orig_slab_uuid = slab_uuid
        # Original Structure
        slab_struct_orig = Slab.from_dict(
            mmdb.db["fireworks"].find_one({"spec.uuid": orig_slab_uuid})["spec"][
                "_tasks"
            ][0]["structure"]
        )
        # Sort
        slab_struct_orig.sort()
        # Strip orig slab object of oxi states to accommodate MXide
        slab_struct_orig.remove_oxidation_states()
        slab_struct_orig.oriented_unit_cell.remove_oxidation_states()

        # try:
        #    orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})[
        #        "orig_inputs"
        #    ]["incar"]["MAGMOM"]
        # except KeyError:  # Seems like the schema changes when fake_vasp on?
        #    orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})["input"][
        #        "incar"
        #    ]["MAGMOM"]
        orig_magmoms = slab_struct_orig.site_properties["magmom"]
        orig_site_properties = slab_struct.site_properties
        # Replace the magmoms with the initial values
        orig_site_properties["magmom"] = orig_magmoms
        slab_struct = slab_struct.copy(site_properties=orig_site_properties)
        slab_struct.add_site_property("bulk_wyckoff", slab_wyckoffs)
        slab_struct.add_site_property("bulk_equivalent", slab_equivalents)
        slab_struct.add_site_property("forces", slab_forces)
        # Original Structure site decoration
        slab_struct_orig = slab_struct_orig.copy(site_properties=orig_site_properties)
        slab_struct_orig.add_site_property("bulk_wyckoff", slab_wyckoffs)
        slab_struct_orig.add_site_property("bulk_equivalent", slab_equivalents)

        # Because we are only interested in the surface, we don't need the relaxed
        # OUC so we will just get it from the slab metadata from the slab_uuid
        ouc = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"]["oriented_unit_cell"]
        )

        # # Oriented unit cell site properties
        # oriented_wyckoffs = [
        #     site["properties"]["bulk_wyckoff"]
        #     for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
        #         "oriented_unit_cell"
        #     ]["sites"]
        # ]
        # oriented_equivalents = [
        #     site["properties"]["bulk_equivalent"]
        #     for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
        #         "oriented_unit_cell"
        #     ]["sites"]
        # ]

        # # Decorate oriented unit cell with site properties
        # ouc.add_site_property("bulk_wyckoff", oriented_wyckoffs)
        # ouc.add_site_property("bulk_equivalent", oriented_equivalents)

        # oriented_struct_orig.add_site_property("bulk_wyckoff", oriented_wyckoffs)
        # oriented_struct_orig.add_site_property("bulk_equivalent", oriented_equivalents)

        # Optimized Slab object
        # Output
        optimized_slab = Slab(
            slab_struct.lattice,
            slab_struct.species,
            slab_struct.frac_coords,
            miller_index=list(map(int, miller_index)),
            oriented_unit_cell=ouc,
            shift=0,
            scale_factor=0,
            energy=mmdb.db["tasks"].find_one({"uuid": slab_uuid})["output"]["energy"],
            site_properties=slab_struct.site_properties,
        )
        # Input
        original_slab = Slab(
            slab_struct_orig.lattice,
            slab_struct_orig.species,
            slab_struct_orig.frac_coords,
            miller_index=list(map(int, miller_index)),
            oriented_unit_cell=ouc,
            shift=0,
            scale_factor=0,
            energy=0,
            site_properties=slab_struct_orig.site_properties,
        )

        # Now we create the adslabs across all rotations of the adsorbate for a specific U value
        adslab_fws = []
        for adsorbate in adsorbates:

            # Place a single adsorbate using MXide

            mxidegen = MXideAdsorbateGenerator(
                original_slab,
                repeat=[1, 1, 1],
                verbose=False,
                positions=["MX_adsites"],  # tol=1.59,# relax_tol=0.025
            )
            bulk_like, _ = mxidegen.get_bulk_like_adsites()
            #            adslabs, bulk_like_shifted = get_clockwise_rotations(
            #                original_slab, optimized_slab, adsorbate
            #            )
            # Bondlength and X
            _, X = mxidegen.bondlengths_dict, mxidegen.X
            bulk_like_shifted = _bulk_like_adsites_perturbation(
                original_slab, optimized_slab, bulk_like, X=X
            )
            # Choose random site
            if adsite_index is None:
                bulk_like_site_index = np.random.choice(
                    np.arange(len(bulk_like_shifted))
                )
            else:
                bulk_like_site_index = adsite_index
            bulk_like_site = bulk_like_shifted[bulk_like_site_index]
            adslab = add_adsorbates(optimized_slab, [bulk_like_site], adsorbate)
            adslab.sort()
            elements = [el.name for el in adslab.composition.elements]
            # Assume that the adsorbate will not require +U and get the values from the U_values dict
            UU = [U_values[el] if el in U_values else 0 for el in elements]
            UL = [2 if el in U_values else 0 for el in elements]
            UJ = [0 for el in elements]
            ads_vis = MOSurfaceSet(adslab, UU=UU, UJ=UJ, UL=UL, apply_U=True)
            name = f"{adslab.composition.reduced_formula}-{miller_index}_{UU}"
            ads_slab_uuid = str(uuid.uuid4())
            ads_slab_fw = AdsSlab_FW(
                adslab,
                name=name,
                slab_uuid=slab_uuid,
                ads_slab_uuid=ads_slab_uuid,
                vasp_cmd=VASP_CMD,
                db_file=db_file,
                run_fake=False,
                vasp_input_set=ads_vis,  # Need a different U setting to accommodate the adsorbate
            )
            adslab_fws.append(ads_slab_fw)

        # Spawn the new adslab fws and the post-processing/analysis workflow through FWAction
        # The post-processing task needs to be a child for all the adslab_fws as part of a workflow
        # Then we trigger an action, which is to create the workflow
        # TODO: Post-processing goes here
        return FWAction(detours=adslab_fws)
