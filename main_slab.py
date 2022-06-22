"""
Copyright (c) 2022 Carnegie Mellon University.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


from WhereWulff.launchers.slabflows import SlabFlows

# Import CIF file
cif_file = "./benchmark/mp_oxide_cifs/IrO2_136.cif"

# WhereWulff method and config
cat_flows = SlabFlows(
    cif_file,
    exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)],  # only 110 and 101
    run_fake=True,
    metal_site="Ir",
)

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="fireworks",
    port=27017,
    username="",
    password="",
)
