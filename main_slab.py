from CatFlows.launchers.catflows import CatFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
cif_file = "./benchmark/mp_oxide_cifs/IrO2_136.cif"

# CatFlows method and config
cat_flows = CatFlows(
    cif_file,
    exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)],
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
