from pymatgen.core import surface
from fireworks import Firework

from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure

from CatFlows.analysis.oer import OER_SingleSiteAnalyzer


def OER_SingleSiteAnalyzer_FW(
    reduced_formula="",
    name="",
    miller_index="",
    metal_site="",
    slab_uuid="",
    ads_slab_uuids="",
    surface_termination="",
    parents=None,
    db_file=DB_FILE,
):
    """
    Converts the OER_SingleSiteAnalyzer FireTask to FireWorks.
    """

    # FW
    fw = Firework(
        OER_SingleSiteAnalyzer(
            reduced_formula=reduced_formula,
            miller_index=miller_index,
            metal_site=metal_site,
            slab_uuid=slab_uuid,
            ads_slab_uuids=ads_slab_uuids,
            surface_termination=surface_termination,
            db_file=db_file,
            to_db=True,
        ),
        name=name,
        parents=parents,
    )

    return fw
