from __future__ import absolute_import, division, print_function, unicode_literals

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.firetasks.slab_ads import SlabAdsFireTask


def SlabAds_WF(
    bulk_structure, adsorbates, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE
):
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
    ads_slab_fw = Firework(
        SlabAdsFireTask(
            reduced_formula=bulk_formula,
            db_file=db_file,
            adsorbates=adsorbates,
            slabs=None,
            vasp_cmd=vasp_cmd,
        ),
        name=f"{bulk_formula} Ads_slab optimization",
        parents=parents,
    )

    all_fws = [ads_slab_fw]
    if parents is not None:
        all_fws.extend(parents)
    ads_wf = Workflow(all_fws, name="{} Ads_slab optimizations".format(bulk_formula))
    return ads_wf, all_fws
