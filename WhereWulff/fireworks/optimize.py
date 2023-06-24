from typing import Counter
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure, env_chk
from atomate.vasp.firetasks.run_calc import RunVaspFake, RunVaspDirect, RunVaspCustodian
import os

from WhereWulff.dft_settings.settings import MOSurfaceSet
from WhereWulff.firetasks.handlers import ContinueOptimizeFW


# Dictionary that holds the paths to the VASP input
# and output files. Right now this assumes that the code is run
# in a container (/home/jovyan) with the files placed in the right folder.
# Maps fw_name to the ref_dir
ref_dirs = {
    # RuO2
    "RuO2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_bulk_110",
    "RuO2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_bulk_101",
    "RuO2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_slab_110",
    "RuO2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_slab_101",
    "RuO2-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_Ox_pbx_1",
    "RuO2-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_1",
    "RuO2-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_2",
    "RuO2-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_3",
    "RuO2-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_4",
    "RuO2-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_Ox_pbx_1",
    "RuO2-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_1",
    "RuO2-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_2",
    "RuO2-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_3",
    "RuO2-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_4",
    "RuO2-110-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_reference",
    "RuO2-110-Ru-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_0",
    "RuO2-110-Ru-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_1",
    "RuO2-110-Ru-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_2",
    "RuO2-110-Ru-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_3",
    "RuO2-110-Ru-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_0",
    "RuO2-110-Ru-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_1",
    "RuO2-110-Ru-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_2",
    "RuO2-110-Ru-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_3",
    "RuO2-110-Ru-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_0",
    "RuO2-110-Ru-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_1",
    "RuO2-110-Ru-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_2",
    "RuO2-110-Ru-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_3",
    "RuO2-101-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_reference",
    "RuO2-101-Ru-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_0",
    "RuO2-101-Ru-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_1",
    "RuO2-101-Ru-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_2",
    "RuO2-101-Ru-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_3",
    "RuO2-101-Ru-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_0",
    "RuO2-101-Ru-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_1",
    "RuO2-101-Ru-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_2",
    "RuO2-101-Ru-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_3",
    "RuO2-101-Ru-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_0",
    "RuO2-101-Ru-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_1",
    "RuO2-101-Ru-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_2",
    "RuO2-101-Ru-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_3",
    # IrO2
    "IrO2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_bulk_110",
    "IrO2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_bulk_101",
    "IrO2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_slab_110",
    "IrO2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_slab_101",
    "IrO2-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_Ox_pbx_1",
    "IrO2-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_1",
    "IrO2-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_2",
    "IrO2-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_3",
    "IrO2-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_4",
    "IrO2-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_Ox_pbx_1",
    "IrO2-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_1",
    "IrO2-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_2",
    "IrO2-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_3",
    "IrO2-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_4",
    "IrO2-110-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_reference",
    "IrO2-110-Ir-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_0",
    "IrO2-110-Ir-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_1",
    "IrO2-110-Ir-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_2",
    "IrO2-110-Ir-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_3",
    "IrO2-110-Ir-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_0",
    "IrO2-110-Ir-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_1",
    "IrO2-110-Ir-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_2",
    "IrO2-110-Ir-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_3",
    "IrO2-110-Ir-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_0",
    "IrO2-110-Ir-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_1",
    "IrO2-110-Ir-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_2",
    "IrO2-110-Ir-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_3",
    "IrO2-101-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_reference",
    "IrO2-101-Ir-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_0",
    "IrO2-101-Ir-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_1",
    "IrO2-101-Ir-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_2",
    "IrO2-101-Ir-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_3",
    "IrO2-101-Ir-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_0",
    "IrO2-101-Ir-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_1",
    "IrO2-101-Ir-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_2",
    "IrO2-101-Ir-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_3",
    "IrO2-101-Ir-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_0",
    "IrO2-101-Ir-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_1",
    "IrO2-101-Ir-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_2",
    "IrO2-101-Ir-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_3",
    # RuCrO4
    "CrRuO4_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_bulk",
    "CrRuO4_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_bulk",
    "CrRuO4_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_slab",
    "CrRuO4_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_slab",
    "CrRuO4-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_1",
    "CrRuO4-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_2",
    "CrRuO4-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_3",
    "CrRuO4-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_4",
    "CrRuO4-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_Ox_1",
    "CrRuO4-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_1",
    "CrRuO4-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_2",
    "CrRuO4-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_3",
    "CrRuO4-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_4",
    "CrRuO4-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_Ox_1",
    # RuTiO4
    "TiRuO4_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_bulk",
    "TiRuO4_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_bulk",
    "TiRuO4_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_slab",
    "TiRuO4_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_slab",
    "TiRuO4-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_1",
    "TiRuO4-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_2",
    "TiRuO4-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_3",
    "TiRuO4-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_4",
    "TiRuO4-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_Ox_1",
    "TiRuO4-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_1",
    "TiRuO4-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_2",
    "TiRuO4-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_3",
    "TiRuO4-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_4",
    "TiRuO4-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_Ox_1",
    # TiCrRu2Ox - 101
    "TiCr(RuO4)2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_bulk",
    "Ti9Cr11(RuO4)20_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_slab",
    "Ti9Cr11(RuO4)20-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_1",
    "Ti9Cr11(RuO4)20-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_2",
    "Ti9Cr11(RuO4)20-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_3",
    "Ti9Cr11(RuO4)20-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_4",
    "Ti9Cr11(RuO4)20-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_O_1",
    # BaSrCo2O6 - 001 - OH terminated
    "BaSr(CoO3)2_001 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_bulk",
    "Ba5Sr5(Co6O17)2_001 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_slab",
    "Ba5Sr5(Co6O17)2-001-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_1",
    "Ba5Sr5(Co6O17)2-001-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_2",
    "Ba5Sr5(Co6O17)2-001-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_3",
    "Ba5Sr5(Co6O17)2-001-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_4",
    "Ba5Sr5(Co6O17)2-001-O_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_O_1",
    # Ba5Sr5(Co5O16)2 - 101
    "BaSr(CoO3)2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_101_results/BaSrCoO_101_bulk",
    "Ba5Sr5(Co5O16)2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_101_results/BaSrCoO_101_slab",
    # Ba5Ti10Sn5O32 - 101
    "BaTi2SnO6_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_101_results/BaSnTiO_101_bulk",
    "Ba5Ti10Sn5O32_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_101_results/BaSnTiO_101_slab",
    # BaTi2SnO6 - 110
    "BaTi2SnO6_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_bulk",
    "BaTi2SnO6_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_slab",
    "BaTi2SnO6-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_4",
    "BaTi2SnO6-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_3",
    "BaTi2SnO6-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_2",
    "BaTi2SnO6-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_1",
    "BaTi2SnO6-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_Ox",
    # FeSb2O6 - 110
    "Fe(SbO3)2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/FeSbOx_110_results/FeSbOx_101_bulk",
    "Fe(SbO3)2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/FeSbOx_110_results/FeSbOx_101_slab",
    # Au - 100
    "Au_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_bulk",
    "Au_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_slab",
    # Pt - 110
    "Pt_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_bulk",
    "Pt_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_slab",
    # Pt - 100
    "Pt_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_bulk",
    "Pt_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_slab",
    # Pt - 111
    "Pt_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_bulk",
    "Pt_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_slab",
    # Pt - 211
    "Pt_211 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_211_results/Pt_211_bulk",
    "Pt_211 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_211_results/Pt_211_slab",
    # Au - 110
    "Au_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_bulk",
    "Au_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_slab",
    # Ag - 110
    "Ag_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_bulk",
    "Ag_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_slab",
    # Ir - 110
    "Ir_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_bulk",
    "Ir_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_slab",
    # Pd - 110
    "Pd_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_bulk",
    "Pd_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_slab",
    # Pd - 100
    "Pd_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_bulk",
    "Pd_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_slab",
    "Pd-100-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_OH_1",
    "Pd-100-O_1": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_O_1",
    "Pd-100-Pd-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-reference",
    "Pd-100-Pd-OOH_4": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-OOH_4",
    "Pd-100-Pd-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-OH_3",
}


def Bulk_FW(
    bulk,
    name="",
    vasp_input_set=None,
    parents=None,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a bulk firework. Returns an OptimizeFW for the specified slab.

    Args:
        bulk              (Struct Object)   : Structure corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to bulk calculation.
    """
    import uuid

    # Generate a unique ID for Bulk_FW
    fw_bulk_uuid = uuid.uuid4()

    # DFT Method
    if not vasp_input_set:
        vasp_input_set = MOSurfaceSet(bulk, bulk=True)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=bulk,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "uuid_lineage": [],
            "_pass_job_info": True,
            "uuid": fw_bulk_uuid,
            "wall_time": wall_time,
            "max_tries": 10,
            "name": name,
            "is_bulk": True,
        },
    )
    if run_fake:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "TiCr(RuO4)2" in name
            or "Co" in name
            or "Ti" in name
            or "Sb" in name
            or "Au" in name
            or "Pt" in name
            or "Ag" in name
            or "Pd" in name
            or "Ir" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # This is for submitting on Perlmutter, where there is an issue between custodian and the compiled vasp version
        # fw.tasks[1] = RunVaspDirect(
        #    vasp_cmd=vasp_cmd
        # )  # We run vasp without custodian (RAW)

        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": 43200})

    # Append Continue-optimizeFW for wall-time handling and use for uuid message
    # passing
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=True, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add bulk_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_bulk_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    return fw


def Slab_FW(
    slab,
    name="",
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)     : Slab corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """
    import uuid

    # Generate a unique ID for Slab_FW
    fw_slab_uuid = uuid.uuid4()

    # DFT Method
    if not vasp_input_set:
        vasp_input_set = MOSurfaceSet(slab, bulk=False)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=slab,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "uuid_lineage": [],
            "_pass_job_info": True,
            "uuid": fw_slab_uuid,
            "wall_time": wall_time,
            "max_tries": 10,
            "name": name,
            "is_bulk": False,
        },
    )
    if run_fake:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "TiCr(RuO4)2" in name
            or "Co" in name
            or "Ti" in name
            or "Sb" in name
            or "Au" in name
            or "Pt" in name
            or "Ag" in name
            or "Pd" in name
            or "Ir" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # fw.tasks[1] = RunVaspDirect(vasp_cmd=vasp_cmd)
        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_slab_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw


def AdsSlab_FW(
    slab,
    name="",
    oriented_uuid="",
    slab_uuid="",
    ads_slab_uuid="",
    is_adslab=True,
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a ads_slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)     : Slab corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """

    # DFT Method
    if not vasp_input_set:
        vasp_input_set = MOSurfaceSet(slab, bulk=False)
    # breakpoint()

    # FW
    fw = OptimizeFW(
        name=name + "_gpu",
        structure=slab,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "_pass_job_info": True,
            "uuid": ads_slab_uuid,
            "uuid_lineage": [],
            "wall_time": wall_time,
            "name": name,
            "max_tries": 10,
            "is_bulk": False,
            "is_adslab": is_adslab,
            "oriented_uuid": oriented_uuid,  # adslab FW should get terminal node ids
            "slab_uuid": slab_uuid,
            "is_bulk": False,
        },
    )
    if run_fake and "-Ru-" not in name and "-Co-" not in name and "-Ti-" not in name:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "Co" in name
            or "Ti" in name
            or "Pd" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        fw.tasks[1] = RunVaspCustodian(vasp_cmd=vasp_cmd)
        # Switch-off GzipDir for WAVECAR transferring
        # fw.tasks[1] = RunVaspDirect(vasp_cmd=vasp_cmd)
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": ads_slab_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw
