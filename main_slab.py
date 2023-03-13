"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from WhereWulff.launchers.slabflows import SlabFlows


# Import CIF file
cif_file = "./BaSrCo2O6_bulk.cif"

# CatFlows method and config
cat_flows = SlabFlows(
    cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 0, 1)]
)  # Only do 001 for this example

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="<<DB-NAME>>",
    port="<<DB-PORT>>",
    username="<<DB-USERNAME>>",
    password="<<DB-PASSWORD>>",
)
