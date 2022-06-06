import uuid
import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE
from CatFlows.workflows.surface_pourbaix import get_clockwise_rotations


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
        "U_value",
        "vis",
    ]

    def run_task(self, fw_spec):
        vis = self["vis"]
        adsorbates = self["adsorbates"]
        U_value = self["U_value"]
        reduced_formula = self["reduced_formula"]
        miller_index = self["miller_index"]
        bulk_slab_key = "_".join([reduced_formula, miller_index])
        # Get the slab uuid
        slab_uuid = fw_spec.get(bulk_slab_key)["slab_uuid"]
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
        orig_slab_uuid = mmdb.db["fireworks"].find_one({"spec.uuid": slab_uuid})[
            "spec"
        ]["uuid_lineage"][0]
        # Original Structure
        slab_struct_orig = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})["input"]["structure"]
        )
        try:
            orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})[
                "orig_inputs"
            ]["incar"]["MAGMOM"]
        except KeyError:  # Seems like the schema changes when fake_vasp on?
            orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})["input"][
                "incar"
            ]["MAGMOM"]
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

        # Oriented unit cell Structure output and input
        orient_struct = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": oriented_uuid})["output"]["structure"]
        )
        oriented_struct_orig = Structure.from_dict(
            mmdb.db["tasks"].find_one({"uuid": oriented_uuid})["input"]["structure"]
        )

        # Oriented unit cell site properties
        oriented_wyckoffs = [
            site["properties"]["bulk_wyckoff"]
            for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
                "oriented_unit_cell"
            ]["sites"]
        ]
        oriented_equivalents = [
            site["properties"]["bulk_equivalent"]
            for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
                "oriented_unit_cell"
            ]["sites"]
        ]

        # Decorate oriented unit cell with site properties
        orient_struct.add_site_property("bulk_wyckoff", oriented_wyckoffs)
        orient_struct.add_site_property("bulk_equivalent", oriented_equivalents)

        oriented_struct_orig.add_site_property("bulk_wyckoff", oriented_wyckoffs)
        oriented_struct_orig.add_site_property("bulk_equivalent", oriented_equivalents)

        # Optimized Slab object
        # Output
        optimized_slab = Slab(
            slab_struct.lattice,
            slab_struct.species,
            slab_struct.frac_coords,
            miller_index=list(map(int, miller_index)),
            oriented_unit_cell=orient_struct,
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
            oriented_unit_cell=oriented_struct_orig,
            shift=0,
            scale_factor=0,
            energy=0,
            site_properties=slab_struct_orig.site_properties,
        )

        # Now we create the adslabs across all rotations of the adsorbate for a specific U value
        adslab_fws = []
        for adsorbate in adsorbates:
            adslabs, bulk_like_shifted = get_clockwise_rotations(
                original_slab, optimized_slab, adsorbate
            )
            for adslab_label, adslab in adslabs.items():
                name = f"{optimized_slab.composition.reduced_formula}-{miller_index}-{adslab_label}"
                ads_slab_uuid = str(uuid.uuid4())
                ads_slab_fw = AdsSlab_FW(
                    adslab,
                    name=name,
                    ads_slab_uuid=ads_slab_uuid,
                    vasp_cmd=vasp_cmd,
                    db_file=db_file,
                    run_fake=False,
                    vasp_input_set=vis,
                )
                adslab_fws.append(ads_slab_fw)

        # Spawn the new adslab fws and the post-processing/analysis workflow through FWAction
        # The post-processing task needs to be a child for all the adslab_fws as part of a workflow
