"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from WhereWulff.firetasks.oer_single_site import OERSingleSiteFireTask


def OER_WF(
    bulk_structure,
    miller_index,
    slab_orig,
    bulk_like_sites,
    ads_dict_orig,
    metal_site,
    applied_potential=1.60,
    applied_pH=0,
    parents=None,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    surface_pbx_uuid="",
):
    """
    Wrap-up workflow to do the OER Single site WNA after SurfacePBX.
    Args:
        bulk_structure (Structure): Bulk structure to refer the wulff shape
        miller_index   (String)   : Crystallographic orientation (h,k,l).
        applied_potential (float) : Potential at which the surface performs OER.
        applied_pH        (int)   : pH conditions for either acidic or alkaline OER.
        parents           (list)  : fw_ids for previous FireTasks.
        vasp_cmd                  : VASP executable
        db_file                   : DB file.
    Returns:
        OER Workflow to generate/optimize OER intermediates and reactivity Analysis.
    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # WulffShape Analysis
    oer_fw = Firework(
        OERSingleSiteFireTask(
            reduced_formula=bulk_formula,
            miller_index=miller_index,
            slab_orig=slab_orig,
            bulk_like_sites=bulk_like_sites,
            ads_dict_orig=ads_dict_orig,
            metal_site=metal_site,
            applied_potential=applied_potential,
            applied_pH=applied_pH,
            db_file=db_file,
            vasp_cmd=vasp_cmd,
            surface_pbx_uuid=surface_pbx_uuid,
        ),
        name=f"{bulk_formula}-{miller_index} OER Single Site WNA",
        parents=parents,
    )

    return oer_fw
