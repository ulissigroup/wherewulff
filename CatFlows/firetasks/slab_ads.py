import uuid
import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.workflows.surface_pourbaix import SurfacePBX_WF


@explicit_serialize
class SlabAdsFireTask(FiretaskBase):
    """
    Slab_Ads OptimizeFW.

    Args:
        reduced_formula:
        slabs          :
        adsorbates     :
        db_file        :
        vasp_cmd       :

    Returns:
        SLAB_ADS Firetasks.

    """

    required_params = [
        "bulk_structure",
        "reduced_formula",
        "slabs",
        "adsorbates",
        "vasp_cmd",
        "db_file",
        "run_fake",
        "metal_site",
        "applied_potential",
        "applied_pH",
    ]
    optional_params = ["_pass_job_info", "_add_launchpad_and_fw_id"]

    def run_task(self, fw_spec):

        # Variables
        bulk_structure = self["bulk_structure"]  # already deserialized
        reduced_formula = self["reduced_formula"]
        slabs = self["slabs"]
        adsorbates = self["adsorbates"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        wulff_uuid = fw_spec.get("wulff_uuid", None)
        run_fake = self.get("run_fake", False)
        metal_site = self.get("metal_site", "")
        applied_potential = self.get("applied_potential", 1.6)
        applied_pH = self.get("applied_pH", 0.0)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        # Slab_Ads
        if slabs is None:
            # Get wulff-shape collection from DB
            if wulff_uuid is not None:
                collection = mmdb.db[f"{reduced_formula}_wulff_shape_analysis"]
                wulff_metadata = collection.find_one(
                    {"task_label": f"{reduced_formula}_wulff_shape_{wulff_uuid}"}
                )

                # Filter by surface contribution
                filtered_slab_miller_indices = [
                    k for k, v in wulff_metadata["area_fractions"].items() if v > 0.0
                ]
                # Create the set of reduced_formulas
                bulk_slab_keys = [k
                    for k in fw_spec
                    if "-" in k
                    and "_" in k
                    and k.split("_")[-1] in filtered_slab_miller_indices]
            else:
                # This is the case where there is no Wulff shape because
                # there is only one miller index
                # Get the bulk_slab_key from the fw_spec
                bulk_slab_keys = [
                    k for k in fw_spec if f"{reduced_formula}" in k
                ]  # FIXME: Need bulk_reduced_formula
                # and slab_reduced_formula to handle case where non-stoichiometric
                filtered_slab_miller_indices = [
                    bsk.split("_")[-1] for bsk in bulk_slab_keys
                ]

            # Re-build PMG Slab object from optimized structures
            slab_candidates = []
            for miller_index, bulk_slab_key in zip(
                filtered_slab_miller_indices, bulk_slab_keys
            ):
                # Retrieve the oriented_uuid and the slab_uuid for the surface orientation
                oriented_uuid = fw_spec.get(bulk_slab_key)["oriented_uuid"]
                slab_uuid = fw_spec.get(bulk_slab_key)["slab_uuid"]
                slab_wyckoffs = [
                    site["properties"]["bulk_wyckoff"]
                    for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
                        "sites"
                    ]
                ]
                slab_equivalents = [
                    site["properties"]["bulk_equivalent"]
                    for site in mmdb.db["tasks"].find_one({"uuid": slab_uuid})["slab"][
                        "sites"
                    ]
                ]
                slab_forces = mmdb.db["tasks"].find_one({"uuid": slab_uuid})["output"][
                    "forces"
                ]
                slab_struct = Structure.from_dict(
                    mmdb.db["tasks"].find_one({"uuid": slab_uuid})["output"][
                        "structure"
                    ]
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
                    orig_slab_uuid = mmdb.db["fireworks"].find_one(
                        {"spec.uuid": slab_uuid}
                    )["spec"]["uuid_lineage"][0]
                else:
                    orig_slab_uuid = slab_uuid
                # Original Structure
                slab_struct_orig = Slab.from_dict(
                    mmdb.db["fireworks"].find_one({"spec.uuid": orig_slab_uuid})[
                        "spec"
                    ]["_tasks"][0]["structure"]
                )
                # Strip orig slab object of oxi states to accommodate MXide
                slab_struct_orig.remove_oxidation_states()
                slab_struct_orig.oriented_unit_cell.remove_oxidation_states()

                try:
                    orig_magmoms = mmdb.db["tasks"].find_one({"uuid": orig_slab_uuid})[
                        "orig_inputs"
                    ]["incar"]["MAGMOM"]
                except KeyError:  # Seems like the schema changes when fake_vasp on?
                    orig_magmoms = slab_struct_orig.site_properties["magmom"]
                orig_site_properties = slab_struct.site_properties
                # Replace the magmoms with the initial values
                orig_site_properties["magmom"] = orig_magmoms
                slab_struct = slab_struct.copy(site_properties=orig_site_properties)
                slab_struct.add_site_property("bulk_wyckoff", slab_wyckoffs)
                slab_struct.add_site_property("bulk_equivalent", slab_equivalents)
                slab_struct.add_site_property("forces", slab_forces)
                # Original Structure site decoration
                slab_struct_orig = slab_struct_orig.copy(
                    site_properties=orig_site_properties
                )
                slab_struct_orig.add_site_property("bulk_wyckoff", slab_wyckoffs)
                slab_struct_orig.add_site_property("bulk_equivalent", slab_equivalents)

                # Oriented unit cell Structure output and input
                orient_struct = Structure.from_dict(
                    mmdb.db["tasks"].find_one({"uuid": oriented_uuid})["output"][
                        "structure"
                    ]
                )
                oriented_struct_orig = Structure.from_dict(
                    mmdb.db["tasks"].find_one({"uuid": oriented_uuid})["input"][
                        "structure"
                    ]
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

                oriented_struct_orig.add_site_property(
                    "bulk_wyckoff", oriented_wyckoffs
                )
                oriented_struct_orig.add_site_property(
                    "bulk_equivalent", oriented_equivalents
                )

                # Optimized Slab object
                slab_candidates.append(
                    (
                        # Output
                        Slab(
                            slab_struct.lattice,
                            slab_struct.species,
                            slab_struct.frac_coords,
                            miller_index=list(map(int, miller_index)),
                            oriented_unit_cell=orient_struct,
                            shift=0,
                            scale_factor=0,
                            energy=mmdb.db["tasks"].find_one({"uuid": slab_uuid})[
                                "output"
                            ]["energy"],
                            site_properties=slab_struct.site_properties,
                        ),
                        # Input
                        Slab(
                            slab_struct_orig.lattice,
                            slab_struct_orig.species,
                            slab_struct_orig.frac_coords,
                            miller_index=list(map(int, miller_index)),
                            oriented_unit_cell=oriented_struct_orig,
                            shift=0,
                            scale_factor=0,
                            energy=0,
                            site_properties=slab_struct_orig.site_properties,
                        ),
                        oriented_uuid,
                        slab_uuid,
                    )
                )
            # Generate independent WF for OH/Ox terminations + Surface PBX
            hkl_pbx_wfs = []
            for slab_out, slab_inp, oriented_uuid, slab_uuid in slab_candidates:
                hkl_pbx_wf = SurfacePBX_WF(
                    bulk_structure=bulk_structure,
                    slab=slab_out,
                    slab_orig=slab_inp,
                    slab_uuid=slab_uuid,
                    oriented_uuid=oriented_uuid,
                    adsorbates=adsorbates,
                    vasp_cmd=vasp_cmd,
                    db_file=db_file,
                    run_fake=run_fake,
                    metal_site=metal_site,
                    applied_potential=applied_potential,
                    applied_pH=applied_pH,
                )
                hkl_pbx_wfs.append(hkl_pbx_wf)

        return FWAction(detours=hkl_pbx_wfs)
