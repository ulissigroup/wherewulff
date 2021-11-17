from CatFlows.launchers.catflows import CatFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./RuO2_136.cif"

# CatFlows method and config
cat_flows = CatFlows(cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)])

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="mo_stability",
    port=27017,
    username="mo_stability_admin",
    password="ww1www3w3ree22s223eew",
)
