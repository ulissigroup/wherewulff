"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from fireworks import Firework

from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure

from WhereWulff.analysis.surface_pourbaix import SurfacePourbaixDiagramAnalyzer


def SurfacePBX_FW(
    reduced_formula="",
    name="",
    miller_index="",
    slab_uuid="",
    oriented_uuid="",
    ads_slab_uuids="",
    parents=None,
    db_file=DB_FILE,
    surface_pbx_uuid="",
):

    # FW
    fw = Firework(
        SurfacePourbaixDiagramAnalyzer(
            reduced_formula=reduced_formula,
            miller_index=miller_index,
            slab_uuid=slab_uuid,
            oriented_uuid=oriented_uuid,
            ads_slab_uuids=ads_slab_uuids,
            db_file=db_file,
            to_db=True,
            surface_pbx_uuid=surface_pbx_uuid,
        ),
        name=name,
        parents=parents,
    )

    return fw
