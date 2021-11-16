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

    required_params = ["reduced_formula", "slabs", "adsorbates", "db_file", "vasp_cmd"]
    optional_params = ["_pass_job_info", "_add_launchpad_and_fw_id"]

    def run_task(self, fw_spec):

        # Variables
        reduced_formula = self["reduced_formula"]
        slabs = self["slabs"]
        adsorbates = self["adsorbates"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        wulff_uuid = fw_spec.get("wulff_uuid")

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Slab_Ads
        if slabs is None:
            # Get wulff-shape collection from DB
            collection = mmdb.db[f"{reduced_formula}_wulff_shape_analysis"]
            wulff_metadata = collection.find_one(
                {"task_label": f"{reduced_formula}_wulff_shape_{wulff_uuid}"}
            )

            # Filter by surface contribution
            filtered_slab_miller_indices = [
                k for k, v in wulff_metadata["area_fractions"].items() if v > 0.0
            ]

            # Create the set of reduced_formulas
            bulk_slab_keys = [
                "_".join([reduced_formula, miller_index])
                for miller_index in filtered_slab_miller_indices
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
                # Initialize from original magmoms instead of output ones.
                orig_magmoms = mmdb.db["tasks"].find_one({"uuid": slab_uuid})[
                    "orig_inputs"
                ]["incar"]["MAGMOM"]
                orig_site_properties = slab_struct.site_properties
                # Replace the magmoms with the initial values
                orig_site_properties['magmom'] = orig_magmoms
                slab_struct = slab_struct.copy(site_properties=orig_site_properties)
                slab_struct.add_site_property("bulk_wyckoff", slab_wyckoffs)
                slab_struct.add_site_property("bulk_equivalent", slab_equivalents)
                slab_struct.add_site_property("forces", slab_forces)
                orient_struct = Structure.from_dict(
                    mmdb.db["tasks"].find_one({"uuid": oriented_uuid})["output"][
                        "structure"
                    ]
                )
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
                orient_struct.add_site_property("bulk_wyckoff", oriented_wyckoffs)
                orient_struct.add_site_property("bulk_equivalent", oriented_equivalents)
                slab_candidates.append(
                    (
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
                        oriented_uuid,
                        slab_uuid,
                    )
                )
            # Generate independent WF for OH/Ox terminations + Surface PBX
            hkl_pbx_wfs = []
            for slab, oriented_uuid, slab_uuid in slab_candidates:
                hkl_pbx_wf = SurfacePBX_WF(
                    slab=slab,
                    slab_uuid=slab_uuid,
                    oriented_uuid=oriented_uuid,
                    adsorbates=adsorbates,
                    vasp_cmd=vasp_cmd,
                    db_file=db_file,
                )
                hkl_pbx_wfs.append(hkl_pbx_wf)

        return FWAction(detours=hkl_pbx_wfs)


"""
# Generate a set of OptimizeFW additions that will relax all the adslab in parallel
ads_slab_fws = []
for slab, oriented_uuid, slab_uuid in slab_candidates:
    slab_miller_index = "".join(list(map(str, slab.miller_index)))
    hkl_fws, hkl_uuids = [], []
    for adsorbate in adsorbates:
        adslabs = get_clockwise_rotations(slab, adsorbate)
        for adslab_label, adslab in adslabs.items():
            name = f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
            ads_slab_uuid = uuid.uuid4()
            ads_slab_fw = AdsSlab_FW(
                adslab,
                name=name,
                oriented_uuid=oriented_uuid,
                slab_uuid=slab_uuid,
                ads_slab_uuid=ads_slab_uuid,
                vasp_cmd=vasp_cmd,
            )
            ads_slab_fws.append(ads_slab_fw)
            hkl_fws.append(ads_slab_fw)
            hkl_uuids.append(ads_slab_uuid)

    # Surface PBX Diagram for each surface orientation "independent"
    pbx_name = f"Surface-PBX-{slab.composition.reduced_formula}-{slab_miller_index}"
    pbx_fw = SurfacePBX_FW(
        reduced_formula=reduced_formula,
        name=pbx_name,
        miller_index=slab_miller_index,
        slab_uuid=slab_uuid,
        ads_slab_uuids=hkl_uuids,
        parents=hkl_fws,
    )
    ads_slab_fws.append(pbx_fw)
"""
