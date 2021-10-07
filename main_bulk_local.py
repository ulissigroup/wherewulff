from CatFlows.launchers.bulkflows import BulkFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./RuO2_136.cif"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit_local()

# Run
rapidfire(launchpad)
