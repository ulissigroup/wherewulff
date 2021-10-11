from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import DB_FILE

from CatFlows.analysis.bulk_stability import BulkStabilityAnalysis


def StabilityBulk_WF(bulk_structure, parents=None, db_file=DB_FILE):
    """
    Wrap-up workflow to do the Stability Analysis for each magnetic ordering.

    Args:

    Returns:

    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # BulkStabilityAnalsysis for NM, FM and AFM Single-points
    bulk_stability_fw = Firework(
        BulkStabilityAnalysis(
            reduced_formula=bulk_formula, db_file=db_file
        ),
        name=f"{bulk_formula} Bulk Stability Analysis",
        parents=parents,
    )

    all_fws = [bulk_stability_fw]
    if parents is not None:
        all_fws.extend(parents)
    stab_wf = Workflow(all_fws, name=f"{bulk_formula} Bulk Stability Analysis")
    return stab_wf