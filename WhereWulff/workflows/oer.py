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
    run_fake=False,
    surface_pbx_uuid="",
    # streamline=False,
    checkpoint_path=None,
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

    # Filter out fw that are not same miller_index
    #    parents_hkl = [fw for fw in parents if f"PBX-{bulk_formula}-{miller_index}" in fw.name]

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
            run_fake=run_fake,
            surface_pbx_uuid=surface_pbx_uuid,
            # streamline=streamline,
            checkpoint_path=checkpoint_path,
        ),
        name=f"{bulk_formula}-{miller_index} OER Single Site WNA",
        parents=parents,
    )

    #    all_fws = [oer_fw]
    #    if parents is not None:
    #        all_fws.extend(parents)
    #    oer_wf = Workflow(all_fws, name=f"{bulk_formula}-{miller_index} OER Single Site WNA")
    return oer_fw


def OER_WF_new(
    bulk_structure, miller_index_list, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE
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

    # PBX parents
    parents_pbx = [fw for fw in parents if f"PBX-{bulk_formula} in fw.name"]

    # Loop over hkl
    all_fws = []
    for hkl in miller_index_list:
        exclude_pbx_parents = [fw for fw in parents_pbx if f"{hkl}" not in fw.name]
        parents_hkl = list(set(parents) - set(exclude_pbx_parents))
        oer_fw = Firework(
            OERSingleSiteFireTask(
                reduced_formula=bulk_formula,
                miller_index=hkl,
                db_file=db_file,
                vasp_cmd=vasp_cmd,
            ),
            name=f"{bulk_formula}-{hkl} OER Single Site WNA",
            parents=parents_hkl,
        )
        all_fws.append(oer_fw)

    if parents is not None:
        all_fws.extend(parents)
        oer_wf = Workflow(all_fws, name=f"{bulk_formula} OER Single Site WNA")
    return oer_wf
