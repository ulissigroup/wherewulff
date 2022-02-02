from CatFlows.launchers.bulkflows import BulkFlows

from fireworks.core.rocket_launcher import rapidfire
import yaml

# load yml file to dictionary
credentials = yaml.load(open('./credentials.yml'))['db']

# Import CIF file
cif_file = "/global/homes/i/ibenlo/RuTiCrOx_bulks/Ti1Cr1Ru2O8_1.cif"

# BulkFlow method and config
bulk_flow = BulkFlows(cif_file)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname=credentials['hostname'],
    db_name=credentials['db_name'],
    port=credentials['port'],
    username=credentials['username'],
    password=credentials['password'],
)
