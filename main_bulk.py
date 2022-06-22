"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from WhereWulff.launchers.bulkflows import BulkFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "path_to_cif"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname="localhost",
    db_name="fw_oal",
    port=27017,
    username="fw_oal_admin",
    password="gfde223223222rft3",
)
