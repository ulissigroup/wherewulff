from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.firetasks.static_bulk import StaticBulkFireTask


def StaticBulk_WF(bulk_structure, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Wrap-up workflow to do the Static DFT calculation after EOS Fitting.

    Args:

    Returns:

    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # StaticBulk for NM, FM and AFM fittings
    bulk_static_fw = Firework(
        StaticBulkFireTask(reduced_formula=bulk_formula,
                           bulks=None,
                           vasp_cmd=vasp_cmd,
                           db_file=db_file),
        name=f"{bulk_formula} Static_Bulk DFT Energy",
        parents=parents,
    )

    all_fws = [bulk_static_fw]
    if parents is not None:
        all_fws.extend(parents)
    ads_wf = Workflow(all_fws, name=f"{bulk_formula} Static_Bulk DFT Energy")
    return ads_wf