from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.firetasks.oer_single_site import OERSingleSiteFireTask


def OER_WF(
    bulk_structure, miller_index, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE
):
    """
    Wrap-up workflow to do the OER Single site WNA after SurfacePBX.
    Args:
        bulk_structure (Structure): Bulk structure to refer the wulff shape
        vasp_cmd: vasp executable
        db_file: database file.
    Returns:
        JSON file with Wulff Analysis.
    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # WulffShape Analysis
    oer_fw = Firework(
             OERSingleSiteFireTask(
                reduced_formula=bulk_formula,
                miller_index=miller_index,
                db_file=db_file,
                vasp_cmd=vasp_cmd,
        ),
        name=f"{bulk_formula}-{miller_index} OER Single Site WNA",
        parents=parents,
    )

    all_fws = [oer_fw]
    if parents is not None:
        all_fws.extend(parents)
    oer_wf = Workflow(all_fws, name=f"{bulk_formula}-{miller_index} OER Single Site WNA")
    return oer_wf