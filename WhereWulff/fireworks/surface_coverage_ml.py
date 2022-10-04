"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from fireworks import Firework

from atomate.vasp.config import DB_FILE

from WhereWulff.firetasks.surface_coverage_ml import SurfaceCoverageMLFireTask
from WhereWulff.config import MODEL_CHECKPOINT, MODEL_CONFIG

def SurfaceCoverageML_FW(
    slab,
    slab_ref,
    adsorbate,
    name="",
    miller_index="",
    slab_uuid="",
    oriented_uuid="",
    ads_slab_uuids="",
    parents=None,
    db_file=DB_FILE,
    model_checkpoint=MODEL_CHECKPOINT,
    model_config=MODEL_CONFIG
):

    # FW
    fw = Firework(
        SurfaceCoverageMLFireTask(
            slab=slab,
            slab_ref=slab_ref,
            adsorbate=adsorbate,
            miller_index=miller_index,
            slab_uuid=slab_uuid,
            oriented_uuid=oriented_uuid,
            ads_slab_uuids=ads_slab_uuids,
            model_checkpoint=model_checkpoint,
            model_config=model_config,
            db_file=db_file,
        ),
        name=name,
        parents=parents,
    )

    return fw