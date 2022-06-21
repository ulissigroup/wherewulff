"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from CatFlows.launchers.bulkflows import BulkFlows

# Import CIF file
cif_file = "<<YOUR_CIF_FILE>>"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname="localhost",
    db_name="<<DB-NAME>>",
    port="<<DB-PORT>>",
    username="<<DB-USERNAME>>",
    password="<<DB-PASSWORD>>",
)
