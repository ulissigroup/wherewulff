from CatFlows.launchers.bulkflows import BulkFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "/global/homes/i/ibenlo/RuTiCrOx_bulks/Ti1Cr1Ru2O8_1.cif"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname="localhost",
    db_name="ib_mongo",
    port="27017",
    username="ib_mongo_admin",
    password="mypass",
)
