import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize

from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb

from CatFlows.fireworks.optimize import AdsSlab_FW
from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator


# Angles list
def get_angles(n_rotations=4):
    """Get angles like in the past"""
    angles = []
    for i in range(n_rotations):
        deg = (2 * np.pi / n_rotations) * i
        angles.append(deg)
    return angles


def add_adsorbates(adslab, ads_coords, molecule):
    """Add molecule in all ads_coords once"""
    translated_molecule = molecule.copy()
    for ads_site in ads_coords:
        for mol_site in translated_molecule:
            new_coord = ads_site + mol_site.coords
            adslab.append(
                mol_site.specie,
                new_coord,
                coords_are_cartesian=True,
                properties=mol_site.properties,
            )
    return adslab


# Try the clockwise thing again...
def get_clockwise_rotations(slab, molecule):
    """We need to rush function..."""
    # This will be a inner method
    mxidegen = MXideAdsorbateGenerator(
        slab, repeat=[1, 1, 1], verbose=False, positions=["MX_adsites"], relax_tol=0.025
    )
    bulk_like_sites, _ = mxidegen.get_bulk_like_adsites()

    # set n_rotations to 1 if mono-atomic
    n = len(molecule[0]) if type(molecule).__name__ == "list" else len(molecule)
    n_rotations = 1 if n == 1 else 4

    # Angles
    angles = get_angles(n_rotations=n_rotations)

    # Molecule formula
    molecule_comp = molecule.composition.as_dict()
    molecule_formula = "".join(molecule_comp.keys())

    # rotate OH
    molecule_rotations = mxidegen.get_transformed_molecule_MXides(
        molecule, axis=[0, 0, 1], angles_list=angles
    )

    # placement
    adslab_dict = {}
    for rot_idx in range(len(molecule_rotations)):
        slab_ads = slab.copy()
        slab_ads = add_adsorbates(
            slab_ads, bulk_like_sites, molecule_rotations[rot_idx]
        )
        adslab_dict.update({"{}_{}".format(molecule_formula, rot_idx + 1): slab_ads})

    return adslab_dict


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
                orig_magmoms = mmdb.db["tasks"].find_one({"uuid": slab_uuid})["orig_inputs"]["incar"]["MAGMOM"]
                new_sp = slab_struct.site_properties.update({"magmom": orig_magmoms})
                slab_struct = slab_struct.copy(site_properties=new_sp)

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
            # Generate a set of OptimizeFW additions that will relax all the adslab in parallel
            ads_slab_fws = []
            for slab, oriented_uuid, slab_uuid in slab_candidates:
                slab_miller_index = "".join(list(map(str, slab.miller_index)))
                for adsorbate in adsorbates:
                    adslabs = get_clockwise_rotations(slab, adsorbate)
                    for adslab_label, adslab in adslabs.items():
                        name = f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
                        ads_slab_fw = AdsSlab_FW(
                            adslab,
                            name=name,
                            oriented_uuid=oriented_uuid,
                            slab_uuid=slab_uuid,
                            vasp_cmd=vasp_cmd,
                        )
                        ads_slab_fws.append(ads_slab_fw)

        return FWAction(detours=ads_slab_fws)
