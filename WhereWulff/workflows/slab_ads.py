from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from atomate.vasp.config import DB_FILE, VASP_CMD
from fireworks import Firework, Workflow
from WhereWulff.firetasks.slab_ads import SlabAdsFireTask


def SlabAds_WF(
    bulk_structure,
    adsorbates,
    parents=None,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
    metal_site="",
    applied_potential=1.60,
    applied_pH=0,
    streamline=False,
    checkpoint_path=None,
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
            bulk_structure=bulk_structure,
            reduced_formula=bulk_formula,
            adsorbates=adsorbates,
            slabs=None,
            db_file=db_file,
            vasp_cmd=vasp_cmd,
            run_fake=run_fake,
            metal_site=metal_site,
            applied_potential=applied_potential,
            applied_pH=applied_pH,
            streamline=streamline,
            checkpoint_path=checkpoint_path,
        ),
        name=f"{bulk_formula} Ads_slab optimization",
        parents=parents,
    )

    all_fws = [ads_slab_fw]
    if parents is not None:
        all_fws.extend(parents)
    ads_wf = Workflow(all_fws, name="{} Ads_slab optimizations".format(bulk_formula))
    return ads_wf, all_fws
