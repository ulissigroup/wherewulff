from pymatgen.core import Structure, Molecule
from pymatgen.core.surface import Slab
from pymongo import MongoClient
from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator
from CatFlows.workflows.surface_pourbaix import (
    add_adsorbates,
    get_angles,
    get_clockwise_rotations,
    _bulk_like_adsites_perturbation,
)
from u_effect import analyzeUEffect
from CatFlows.dft_settings.settings import MOSurfaceSet
import numpy as np
import os
from fireworks import LaunchPad, Workflow
from atomate.vasp.fireworks.core import OptimizeFW
from CatFlows.firetasks.handlers import ContinueOptimizeFW
from CatFlows.fireworks.optimize import Slab_FW, AdsSlab_FW
from atomate.vasp.config import VASP_CMD, DB_FILE
import uuid
from atomate.utils.utils import get_meta_from_structure
import itertools
import json
from CatFlows.adsorption.adsorbate_configs import OH
from adslab_with_U import OptimizeAdslabsWithU
from fireworks import Firework

# Connect to the database
client = MongoClient("mongodb://fw_oal_admin:gfde223223222rft3@localhost:27017/fw_oal")
db = client.get_database("fw_oal")


# Basically redoing the slab_ads firetask with methanol adsorption

bulk_slab_key = "TiO2_101"
# fw_spec = {"TiO2_110": {
# "oriented_uuid": "",
# "slab_uuid": ""
# }
# }
fw_spec = {
    bulk_slab_key: {
        "oriented_uuid": "0084e0e6-943f-44d9-a400-80c3494a0e73",
        "slab_uuid": "c8d1aa55-c6b6-4c6b-ac29-af368f369ad0",
        "orig_slab_uuid": "c8d1aa55-c6b6-4c6b-ac29-af368f369ad0",  # Since it did a child
    }
}

miller_index = bulk_slab_key.split("_")[1]
miller_index = "110"
oriented_uuid = fw_spec.get(bulk_slab_key)["oriented_uuid"]
slab_uuid = fw_spec.get(bulk_slab_key)["slab_uuid"]
slab_wyckoffs = [
    site["properties"]["bulk_wyckoff"]
    for site in db["tasks"].find_one({"uuid": slab_uuid})["slab"]["sites"]
]
slab_equivalents = [
    site["properties"]["bulk_equivalent"]
    for site in db["tasks"].find_one({"uuid": slab_uuid})["slab"]["sites"]
]
slab_forces = db["tasks"].find_one({"uuid": slab_uuid})["output"]["forces"]
slab_struct_orig = Slab.from_dict(
    #    db["tasks"].find_one({"uuid": slab_uuid})["input"][
    db["fireworks"].find_one(
        {"spec.uuid": fw_spec.get(bulk_slab_key)["orig_slab_uuid"], "spec.counter": 0}
    )["spec"]["_tasks"][0]["structure"]
    #        "structure"
    #    ]
)


# Load from the json file
# slab_struct_orig = Slab.from_dict(json.loads(open('pristine_perovskite_slab.json').read()))


slab_struct_orig.to(filename="POSCAR_slab_struct_orig")
slab_struct = Structure.from_dict(
    db["tasks"].find_one({"uuid": slab_uuid})["output"]["structure"]
)
orig_magmoms = db["tasks"].find_one({"uuid": slab_uuid})["input"]["incar"][
    "MAGMOM"
]  # FIXME: Sometimes "input" and other times "orig_inputs" in schema??
orig_site_properties = slab_struct.site_properties
# Replace the magmoms with the initial values
orig_site_properties["magmom"] = orig_magmoms
# Original Structure site decoration
slab_struct_orig = slab_struct_orig.copy(site_properties=orig_site_properties)
slab_struct_orig.add_site_property("bulk_wyckoff", slab_wyckoffs)
slab_struct_orig.add_site_property("bulk_equivalent", slab_equivalents)

slab_struct = slab_struct.copy(site_properties=orig_site_properties)
slab_struct.add_site_property("bulk_wyckoff", slab_wyckoffs)
slab_struct.add_site_property("bulk_equivalent", slab_equivalents)
slab_struct.add_site_property("forces", slab_forces)
orient_struct = Structure.from_dict(
    db["tasks"].find_one({"uuid": oriented_uuid})["output"]["structure"]
)
oriented_struct_orig = Structure.from_dict(
    db["tasks"].find_one({"uuid": oriented_uuid})["input"]["structure"]
)
oriented_wyckoffs = [
    site["properties"]["bulk_wyckoff"]
    for site in db["tasks"].find_one({"uuid": slab_uuid})["slab"]["oriented_unit_cell"][
        "sites"
    ]
]
oriented_equivalents = [
    site["properties"]["bulk_equivalent"]
    for site in db["tasks"].find_one({"uuid": slab_uuid})["slab"]["oriented_unit_cell"][
        "sites"
    ]
]
oriented_struct_orig.add_site_property("bulk_wyckoff", oriented_wyckoffs)
oriented_struct_orig.add_site_property("bulk_equivalent", oriented_equivalents)
orient_struct.add_site_property("bulk_wyckoff", oriented_wyckoffs)
orient_struct.add_site_property("bulk_equivalent", oriented_equivalents)
pristine_slab = Slab(
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
pristine_slab.to(filename="POSCAR_TiO2_pristine")
relaxed_slab = Slab(
    slab_struct.lattice,
    slab_struct.species,
    slab_struct.frac_coords,
    miller_index=list(map(int, miller_index)),
    oriented_unit_cell=orient_struct,
    shift=0,
    scale_factor=0,
    energy=db["tasks"].find_one({"uuid": slab_uuid})["output"]["energy"],
    site_properties=slab_struct.site_properties,
)
relaxed_slab.to(filename="POSCAR_TiO2_relaxed")

# Find the adsites on the relaxed slab

# mxidegen = MXideAdsorbateGenerator(
# relaxed_slab, repeat=[1, 1, 1], verbose=False, positions=["MX_adsites"], relax_tol=0.025
# )
# bulk_like_sites, _ = mxidegen.get_bulk_like_adsites()
# Take the relaxed methanol molecule and adsorb it onto the bulk_like_sites (for now full monolayer)
#
## Strip things of oxidation states since MXide cannot parse
pristine_slab = slab_struct_orig.copy()  # FIXME: To remove
pristine_slab.remove_oxidation_states()
pristine_slab.oriented_unit_cell.remove_oxidation_states()
#

struct_meth = Structure.from_file("POSCAR_methanol_vasp_opt")
# Recenter the molecule to place the oxygen at the origin and recast the structure as a molecule
cart_coords = struct_meth.cart_coords
center_cart_coords = cart_coords - cart_coords[-1]
species = [x.symbol for x in struct_meth.species]
mol_meth = Molecule(species, center_cart_coords)
mol_meth.add_site_property("binding_site", [False, False, False, False, False, True])


# adslab = add_adsorbates(relaxed_slab, [bulk_like_sites[0]], mol_meth, z_offset=[0,0,0])
# adslab.to(filename="POSCAR_single_site_adslab_meth")
# This will be a inner method
# with open('pristine_perovskite_slab.json', 'w') as f:
#    json.dump(pristine_slab.as_dict(), f, indent=4)
mxidegen = MXideAdsorbateGenerator(
    pristine_slab,
    repeat=[1, 1, 1],
    verbose=False,
    positions=["MX_adsites"],  # tol=1.59,# relax_tol=0.025
)
bulk_like, _ = mxidegen.get_bulk_like_adsites()


# bulk_like_sites = mxidegen._filter_clashed_sites(bulk_like)

# Bondlength and X
_, X = mxidegen.bondlengths_dict, mxidegen.X
bulk_like_shifted = _bulk_like_adsites_perturbation(
    pristine_slab, relaxed_slab, bulk_like, X=X
)  # FIXME: Change back to the relaxed

# set n_rotations to 1 if mono-atomic
n = len(mol_meth[0]) if type(mol_meth).__name__ == "list" else len(mol_meth)
n_rotations = 1 if n == 1 else 4

# Angles
angles = get_angles(n_rotations=n_rotations)

# Molecule formula
molecule_comp = mol_meth.composition.as_dict()
molecule_formula = "".join(molecule_comp.keys())

# rotate adsorbate
molecule_rotations = mxidegen.get_transformed_molecule_MXides(
    mol_meth, axis=[0, 0, 1], angles_list=angles
)
# placement
adslab_dict = {}
fws = []
Us = np.linspace(0, 6, 3)
launchpad = LaunchPad(
    host="localhost",
    name="fw_oal",
    port=27017,
    username="fw_oal_admin",
    password="gfde223223222rft3",
)

# for coverage in range(len(bulk_like_shifted), len(bulk_like_shifted) + 1):
for coverage in range(1, 2):
    # Mutate the bulk_like_sites based on the coverage
    #    indices_pick = []
    #    while len(indices_pick) < coverage:
    # generate new index
    #        random_index = np.random.choice(np.arange(coverage))
    #        while random_index in indices_pick:
    #            random_index = np.random.choice(np.arange(coverage))
    #        indices_pick.append(random_index)
    sites_pick = itertools.combinations(bulk_like_shifted, coverage)
    for cov_idx, cov_bulk_like_sites in enumerate(list(sites_pick)[:1]):
        # cov_bulk_like_sites = [bulk_like_shifted[i] for i in indices_pick]
        #        for rot_idx in range(len(molecule_rotations)):
        adslab_fws = []
        for U in Us:
            #            slab_ads = relaxed_slab.copy()
            #            slab_ads = add_adsorbates(
            #                slab_ads, list(cov_bulk_like_sites), molecule_rotations[rot_idx]
            #            )
            #            miller_index_str = "".join(list(map(str, relaxed_slab.miller_index)))
            #            name = f"{relaxed_slab.composition.reduced_formula}_{miller_index_str}_cov:{coverage}_{cov_idx}_{molecule_formula}_{rot_idx + 1}"
            #            adslab_dict.update({name: slab_ads})
            #            dir_name = f"{relaxed_slab.composition.reduced_formula}_meth_coverage_{miller_index_str}_new"
            #            if not os.path.exists(dir_name):
            #                os.makedirs(dir_name)
            #            slab_ads.to(filename=f"./{dir_name}/POSCAR_{name}")
            # Send an OptimizeFW calc to the hosted MongoDB for execution
            block_dict = {"s": 0, "p": 1, "d": 2, "f": 3}
            lmaxmix_dict = {"p": 2, "d": 4, "f": 6}
            elements = [el.name for el in pristine_slab.composition.elements]
            blocks = {s.species_string: s.specie.block for s in pristine_slab}
            #            U_values = {el: U if el == "Ti" else 0 for el in elements}
            U_values = {"Ti": U}
            vasp_input_set = MOSurfaceSet(
                pristine_slab,
                bulk=False,
                UJ=[0 for el in elements],
                UU=[U_values[el] if el in U_values else 0 for el in elements],
                UL=[ block_dict[blocks[el]] if el in U_values else 0 for el in elements],
                apply_U=True,
                user_incar_settings={
                    #                    "LDAUJ": [0, 0],
                    #                    "LDAUL": [2, 0],
                    "LDAUPRINT": 0,
                    "LDAUTYPE": 2,
                    #                    "LDAUU": [U, 0], # assume applied to d orbitals but can vary
                    "LMAXMIX": lmaxmix_dict[blocks[[k for k in U_values][0]]], # Assume user is varying only on specie at a time
                },
            )
            #            vasp_input_set.incar.update({'LDAUJ': [0, 0]})
            #            vasp_input_set.incar.update({'LDAUU': [U, 0]})
            #            vasp_input_set.incar.update({'LDAUL': [2, 0]})
            #

            # Root node is the Slab at a specific U value
            slab_fw = Slab_FW(pristine_slab, vasp_input_set=vasp_input_set)
            # Create FW that will spawn the adslabs for across all the adsorbates
            adslab_fw = Firework(
                OptimizeAdslabsWithU(
                    reduced_formula=vasp_input_set.structure.composition.reduced_formula,
                    adsorbates=[molecule_rotations[0]],
                    db_file=DB_FILE,
                    miller_index="".join(
                        map(str, vasp_input_set.structure.miller_index)
                    ),
                    U_values=U_values,
                    vis=vasp_input_set,
                ),
                parents=[slab_fw],
                name=f"Adslabs_{U}",
            )
            #            fw_slab_uuid = uuid.uuid4()
            #
            #            fw = OptimizeFW(
            #                name=name,
            #                structure=slab_ads,
            #                max_force_threshold=None,
            #                vasp_input_set=vasp_input_set,
            #                vasp_cmd=VASP_CMD,
            #                db_file=DB_FILE,
            #                parents=None,
            #                job_type="normal",
            #                spec={
            #                    "counter": 0,
            #                    "_add_launchpad_and_fw_id": True,
            #                    "_pass_job_info": True,
            #                    "uuid": fw_slab_uuid,
            #                    "wall_time": 7200,
            #                    "max_tries": 5,
            #                    "name": name,
            #                    "is_bulk": False,
            #                    "oriented_uuid": oriented_uuid,
            #                    "slab_uuid": slab_uuid,
            #                    "is_adslab": True,
            #                },
            #            )
            #            # Switch-off GzipDir for WAVECAR transferring
            #            fw.tasks[1].update({"gzip_output": False})
            #
            #            # Append Continue-optimizeFW for wall-time handling
            #            fw.tasks.append(
            #                ContinueOptimizeFW(
            #                    is_bulk=False, counter=0, db_file=DB_FILE, vasp_cmd=VASP_CMD
            #                )
            #            )
            #
            #            # Add slab_uuid through VaspToDb
            #            fw.tasks[3]["additional_fields"].update({"uuid": fw_slab_uuid})
            #
            #            # Switch-on WalltimeHandler in RunVaspCustodian
            #            fw.tasks[1].update({"wall_time": 7200})
            #            parent_structure_metadata = get_meta_from_structure(
            #                relaxed_slab.oriented_unit_cell
            #            )
            #            fw.tasks[3]["additional_fields"].update(
            #                {
            #                    "slab": relaxed_slab,
            #                    "parent_structure": relaxed_slab.oriented_unit_cell,
            #                    "parent_structure_metadata": parent_structure_metadata,
            #                }
            #            )
            fws.append(slab_fw)
            adslab_fws.append(adslab_fw)
        # Define the analysis FW - for now as a placeholder that just ingests the slab_uuid and adslab_uuid
        analysis_fw = Firework(analyzeUEffect(), parents=adslab_fws)
        fws.extend(adslab_fws)
        fws.append(analysis_fw)
launchpad.add_wf(Workflow(fws, name="Slabs with U range"))
