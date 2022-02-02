from CatFlows.launchers.catflows import CatFlows
from fireworks.core.rocket_launcher import rapidfire

import yaml
credentials = yaml.load(open('./credentials.yml'))['db']

# Import CIF file
cif_file = "/global/cscratch1/sd/ibenlo/mongo_results/block_2021-12-19-20-42-33-847568/launcher_2022-01-09-04-05-08-234423/slab.cif"

# CatFlows method and config
cat_flows = CatFlows(cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)])

# Get Launchpad
launchpad = cat_flows.submit(
    hostname=credentials['hostname'],
    db_name=credentials['db_name'],
    port=credentials['port'],
    username=credentials['username'],
    password=credentials['password'],
)
