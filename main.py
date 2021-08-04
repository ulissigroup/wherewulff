from CatFlows.workflows.catflows import CatFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./RuO2_136.cif"

# CatFlows method and config
cat_flows = CatFlows(cif_file)

# Get Launchpad
launchpad = cat_flows.submit()

#print(launchpad.get_wf_summary_dict(5))

# Run!
rapidfire(launchpad)
