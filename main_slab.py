from CatFlows.launchers.catflows import CatFlows

from fireworks.core.rocket_launcher import rapidfire

# Import CIF file
# cif_file = "/home/jovyan/BaSrCoO6_bulk_stability/Relaxed_FM_bulk_BaSrCo2O6.cif"
# cif_file = "/home/jovyan/Downloads/hongliang_bulk.cif"
# cif_file = "/home/jovyan/4770/RuTiCrOx_7_relaxed.cif"
cif_file = "/home/jovyan/BaSnTi2Co6_bulk_stability/Template_BaSnTiO6.cif"

# CatFlows method and config
cat_flows = CatFlows(
    cif_file,
    #    symmetrize=True,
    #    stop_at_wulff_analysis=True,
    slab_repeat=[2, 1, 1],
    run_fake=True,
    #    conventional_standard=False,
    #    exclude_hkl=[],
    #    #    exclude_hkl=[(1,1,1),(1,1,0),(1,0,1)],
    exclude_hkl=[(1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
    metal_site="Ti",
)

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="fw_oal",
    port=27017,
    username="fw_oal_admin",
    password="gfde223223222rft3",
)
