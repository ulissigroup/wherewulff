from CatFlows.launchers.catflows import CatFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./RuO2_136.cif"

# CatFlows method and config
cat_flows = CatFlows(cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)])

# Get Launchpad Locally
launchpad = cat_flows.submit(
    hostname=None,
    db_name="",
    port="",
    username="",
    password="",
)
