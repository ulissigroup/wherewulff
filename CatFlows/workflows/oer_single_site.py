from __future__ import absolute_import, division, print_function, unicode_literals
import uuid
import numpy as np

from fireworks import Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from pymatgen.core.surface import Slab

from CatFlows.fireworks.optimize import AdsSlab_FW
from CatFlows.fireworks.oer_single_site import OER_SingleSiteAnalyzer_FW


def OERSingleSite_WF(
    oer_dict,
    slab,
    slab_uuid,
    oriented_uuid,
    surface_termination,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
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
        name = f"{general_reduced_formula}-{miller_index}-{oer_inter_label}"
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
        name=f"{general_reduced_formula}-{miller_index}-OER-Analysis",
        slab_uuid=slab_uuid,
        ads_slab_uuids=oer_uuids,
        surface_termination=surface_termination,
        parents=oer_fws,
        db_file=db_file,
    )

    # Create the workflow
    all_fws = oer_fws + [oer_fw]
    oer_single_site = Workflow(
        all_fws, name=f"{general_reduced_formula}-{miller_index}-OER SingleSite"
    )
    return oer_single_site
