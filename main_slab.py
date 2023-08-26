"""
Copyright (c) 2022 Carnegie Mellon University.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


from WhereWulff.launchers.slabflows import SlabFlows

# Import CIF file
# cif_file = "./benchmark/mp_oxide_cifs/IrO2_136.cif"
# cif_file = "/home/jovyan/BaSnTi2Co6_bulk_stability/Template_BaSnTiO6.cif"
cif_file = "perm_RuNbMo/POSCAR_struct_58.cif"


# WhereWulff method and config
cat_flows = SlabFlows(
    cif_file,
    max_index=1,
    slab_repeat=[1, 1, 1],
    symmetrize=False,
    #    exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)],  # only 110 and 101
    exclude_hkl=[
        (1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        #(0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ],  # Only 011
    run_fake=False,
    #streamline=False,
    metal_site="Ru",
)

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="fw_oal",
    port=27017,
    username="fw_oal_admin",
    password="gfde223223222rft3",
)
