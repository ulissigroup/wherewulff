from fireworks import Firework

from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure

from CatFlows.analysis.surface_pourbaix import SurfacePourbaixDiagramAnalyzer


def SurfacePBX_FW(
    reduced_formula="",
    name="",
    miller_index="",
    slab_uuid="",
    ads_slab_uuids="",
    parents=None,
    db_file=DB_FILE,
):

    # FW
    fw = Firework(
        SurfacePourbaixDiagramAnalyzer(
            reduced_formula=reduced_formula,
            miller_index=miller_index,
            slab_uuid=slab_uuid,
            ads_slab_uuids=ads_slab_uuids,
            db_file=db_file,
            to_db=True,
        ),
        name=name,
        parents=parents,
    )

    return fw
