from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.analysis.wulff_shape import WulffShapeFW


def WulffShape_WF(bulk_structure, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Wrap-up workflow to do the Wulff Shape Analysis after MO_SLABS_WF.

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
    wulff_fw = Firework(
        WulffShapeFW(bulk_structure=bulk_structure, db_file=db_file),
        name="{} wulff shape Task".format(bulk_formula),
        parents=parents,
    )

    all_fws = [wulff_fw]
    if parents is not None:
        all_fws.extend(parents)
    wulff_wf = Workflow(all_fws, name="{} wulff shape analysis".format(bulk_formula))
    return wulff_wf
