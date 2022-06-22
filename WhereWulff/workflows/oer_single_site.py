"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import uuid
import numpy as np

from fireworks import Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from pymatgen.core.surface import Slab

from WhereWulff.fireworks.optimize import AdsSlab_FW
from WhereWulff.fireworks.oer_single_site import OER_SingleSiteAnalyzer_FW


def OERSingleSite_WF(
    oer_dict,
    slab,
    metal_site,
    slab_uuid,
    oriented_uuid,
    surface_termination,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    surface_pbx_uuid="",
):
    """
    Wrap-up workflow for OER single site (wna) + Reactivity Analysis

    Args:

    Returns:
        something
    """
    # Empty lists
    oer_fws, oer_uuids = [], []

    # Reduced formula
    general_reduced_formula = slab.composition.reduced_formula
    miller_index = "".join(list(map(str, slab.miller_index)))

    # Loop over OER intermediates
    for oer_inter_label, oer_inter in oer_dict.items():
        oer_intermediate = Slab.from_dict(oer_inter)
        # reduced_formula = oer_intermediate.composition.reduced_formula
        name = (
            f"{general_reduced_formula}-{miller_index}-{metal_site}-{oer_inter_label}"
        )
        oer_inter_uuid = uuid.uuid4()
        oer_inter_fw = AdsSlab_FW(
            oer_intermediate,
            name=name,
            oriented_uuid=oriented_uuid,
            slab_uuid=slab_uuid,
            ads_slab_uuid=oer_inter_uuid,
            vasp_cmd=vasp_cmd,
        )
        oer_fws.append(oer_inter_fw)
        oer_uuids.append(oer_inter_uuid)

    # Reactivity Analysis
    oer_fw = OER_SingleSiteAnalyzer_FW(
        reduced_formula=str(general_reduced_formula),
        miller_index=miller_index,
        metal_site=metal_site,
        name=f"{general_reduced_formula}-{miller_index}-{metal_site}-OER-Analysis",
        slab_uuid=slab_uuid,
        ads_slab_uuids=oer_uuids,
        surface_termination=surface_termination,
        parents=oer_fws,
        db_file=db_file,
        surface_pbx_uuid=surface_pbx_uuid,
    )

    # Create the workflow
    all_fws = oer_fws + [oer_fw]
    oer_single_site = Workflow(
        all_fws,
        name=f"{general_reduced_formula}-{miller_index}-{metal_site}-OER SingleSite",
    )
    return oer_single_site
