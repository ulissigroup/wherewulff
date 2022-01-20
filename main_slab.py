from CatFlows.launchers.catflows import CatFlows
from fireworks.core.rocket_launcher import rapidfire


# Import CIF file
cif_file = "/global/cscratch1/sd/ibenlo/mongo_results/block_2021-12-19-20-42-33-847568/launcher_2022-01-09-04-05-08-234423/slab.cif"

# CatFlows method and config
cat_flows = CatFlows(cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)])

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="mongodb05.nersc.gov",
    db_name="ib_mongo",
    port=27017,
    username="ib_mongo_admin",
    password="m0ngou0mo12",
)
