from CatFlows.launchers.bulkflows import BulkFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./RuO2_136.cif"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname="localhost",
    db_name="mo_stability",
    port=27017,
    username="mo_stability_admin",
    password="ww1www3w3ree22s223eew",
)
