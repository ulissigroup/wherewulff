"""
Copyright (c) 2022 Carnegie Mellon University.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


from WhereWulff.launchers.slabflows import SlabFlows

# Import CIF file
# cif_file = "./benchmark/mp_oxide_cifs/RuO2_136.cif"
# cif_file = "/home/jovyan/BaSnTi2Co6_bulk_stability/Template_BaSnTiO6.cif"
# cif_file = "./FeSb2O6.cif"
# cif_file = "/Users/yurisanspeur/Downloads/Pt.cif"
# cif_file = "POSCAR_RuPtMoOx_relaxed.cif"
# cif_file = "perm/POSCAR_struct_0.cif"
# cif_file = "POSCAR_RuNbMoOx_2009.cif"
cif_file = "POSCAR_RuNbMoOx_1908.cif"
# cif_file = "Au.cif"

# WhereWulff method and config
cat_flows = SlabFlows(
    cif_file,
    add_magmoms=True,
    slab_repeat=[1, 1, 1],
    # slab_repeat=[4, 4, 1],
    selective_dynamics=True,
    max_index=1,
    symmetrize=False,
    # symmetrize=True,
    # exclude_hkl=[(1,1,1), (2,1,1), (1,0,0),(1, 0, 1), (0, 0, 1), (2,2,1), (2,1,0)],  # only 110 and 100
    exclude_hkl=[
        (0, 0, 1),
        (0, 1, 0),
        # (0, 1, 1),
        (1, 0, 1),
        (1, 0, 0),
        (2, 2, 1),
        (2, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
    ],
    # run_fake=False,
    run_fake=False,
    checkpoint_path=">>checkpoint_path<<",
    #conventional_standard=False,
    conventional_standard=True,
    metal_site="Ru",
    # metal_site="Au",
    is_metal=False,
    # is_metal=True,
    applied_potential=1.5,
    # applied_potential=0.8,
)

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="fw_oal",
    port=27017,
    username="fw_oal_admin",
    password="gfde223223222rft3",
)
