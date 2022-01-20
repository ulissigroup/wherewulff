from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity.strain import Deformation

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.fireworks.core import TransmuterFW

from CatFlows.dft_settings.settings import MOSurfaceSet
from CatFlows.fireworks.optimize import Bulk_FW
from CatFlows.analysis.equation_of_states import FitEquationOfStateFW
import uuid


def EOS_WF(
    bulk_structures_dict,
    deformations=None,
    n_deformations=6,
    vasp_input_set=None,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Equation of state workflow that handles optimization + Deformation + EOS_FIT

    Args:


    Return:
        Workflow, which consist in opt + deformations + EOS_FIT
    """
    fws, fws_all = [], []

    # Bulk-optimization settings
    bulk_structure = Structure.from_dict(bulk_structures_dict["NM"])
    vasp_opt = MOSurfaceSet(bulk_structure, user_incar_settings={"ISIF": 3}, bulk=True)

    # Bulk structure optimization
    opt_name = f"{bulk_structure.composition.reduced_formula}_bulk_optimization"
    fws.append(
        Bulk_FW(
            bulk_structure,
            name=opt_name,
            vasp_input_set=vasp_opt,
            vasp_cmd=vasp_cmd,
            db_file=db_file,
        )
    )
    opt_parent = fws[0]
    fws_all.append(fws[0])

    # Deformations
    if not deformations:
        deformations = [
            (np.identity(3) * (1 + x)).tolist() for x in np.linspace(-0.157, 0.157, n_deformations)
        ]
    deformations = [Deformation(defo_mat) for defo_mat in deformations]
    # breakpoint()
    for counter, (mag_ordering, bulk_struct) in enumerate(bulk_structures_dict.items()):
        bulk_struct = Structure.from_dict(bulk_struct)
        vasp_static = MOSurfaceSet(
            bulk_struct, user_incar_settings={"NSW": 0}, bulk=True
        )
        if counter != 0:
            fws = [opt_parent]
        for n, deformation in enumerate(deformations):
            # Create unique uuid for each deformation
            deform_uuid = uuid.uuid4()
            name_deformation = f"{bulk_structure.composition.reduced_formula}_{mag_ordering}_deformation_{n}"
            fw = TransmuterFW(
                copy_vasp_outputs=False,  # default is True
                name=name_deformation,
                structure=bulk_struct,
                transformations=["DeformStructureTransformation"],
                transformation_params=[{"deformation": deformation.tolist()}],
                vasp_input_set=vasp_static,
                parents=opt_parent,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
            )
            # Add deform_uuid to the task doc in the tasks collection
            fw.tasks[3]["additional_fields"].update({"deform_uuid": deform_uuid})
            # Send the deform_uuid to the corresponding EOS FW
            fw.tasks[3].update(
                {"task_fields_to_push": {f"deformation_uuid_{n}": "deform_uuid"}}
            )
            fws.append(fw)
            fws_all.append(fw)

        # Fit EOS task
        fit_parents = fws[1:]
        name_fit_eos = f"{bulk_structure.composition.reduced_formula}_{mag_ordering}_eos_fitting_analysis"
        fw_analysis = Firework(
            FitEquationOfStateFW(magnetic_ordering=mag_ordering, db_file=db_file),
            name=name_fit_eos,
            parents=fit_parents,
        )
        # fws.append(fw_analysis)
        fws_all.append(fw_analysis)

    # breakpoint()
    # Create Workflow
    wf_eos = Workflow(fws_all)
    wf_eos.name = f"{bulk_structure.composition.reduced_formula}_eos_fitting_analysis"
    return wf_eos, fws_all


def BulkOptimize_WF(
    bulk_structure, vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE
):
    """
    Bulk optimization workflow.

    Args:

    Return:
        Workflow, which consist in bulk optimization.
    """
    fws = []

    # Bulk-optimization
    vasp_opt = MOSurfaceSet(bulk_structure, user_incar_settings={"ISIF": 3}, bulk=True)

    # Bulk structure optimization
    name_bulk = f"{bulk_structure.composition.reduced_formula}_bulk_optimization"
    fws.append(
        Bulk_FW(
            bulk_structure,
            name=name_bulk,
            vasp_input_set=vasp_opt,
            vasp_cmd=vasp_cmd,
            db_file=db_file,
        )
    )

    # Create Workflow
    wf_opt = Workflow(fws)
    wf_opt.name = f"{bulk_structure.composition.reduced_formula}_OPT_WF"
    return wf_opt, fws


def EOS_WF_2(
    bulk_structure,
    deformations=None,
    magnetic_ordering="NM",
    vasp_input_set=None,
    parents=None,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Bulk Deformation workflow that handles Deformation + EOS_Fit.

    Args:

    Return:
        Workflow, which consist in Deformation + EOS Fitting
    """
    # fws = [parents[0]]
    fws = []

    # Linspace deformations
    if not deformations:
        deformations = [
            (np.identity(3) * (1 + x)).tolist() for x in np.linspace(-0.157, 0.157, 6)
        ]

    # Deformations
    vasp_static = MOSurfaceSet(
        bulk_structure, user_incar_settings={"NSW": 0}, bulk=True
    )
    deformations = [Deformation(defo_mat) for defo_mat in deformations]
    for n, deformation in enumerate(deformations):
        name_deformation = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_deformation_{n}"
        fw = TransmuterFW(
            name=name_deformation,
            structure=bulk_structure,
            transformations=["DeformStructureTransformation"],
            transformation_params=[{"deformation": deformation.tolist()}],
            vasp_input_set=vasp_static,
            parents=parents[0],
            vasp_cmd=vasp_cmd,
            db_file=db_file,
        )
        fws.append(fw)

    # Fit EOS task
    # parents = fws[1:]
    name_fit_eos = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_eos_fitting_analysis"
    fw_analysis = Firework(
        FitEquationOfStateFW(magnetic_ordering=magnetic_ordering, db_file=db_file),
        name=name_fit_eos,
        parents=parents,
    )
    fws.append(fw_analysis)

    # Include parents
    # if parents is not None:
    #    fws.extend(parents)

    # breakpoint()

    # Create workflow
    wf_eos = Workflow(fws)
    wf_eos.name(
        f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_EOS_WF"
    )
    return wf_eos


def EOS_WF_OLD(
    bulk_structure,
    deformations=None,
    magnetic_ordering="NM",
    vasp_input_set=None,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Equation of state workflow that handles optimization + Deformation + EOS_FIT

    Args:
        bulk_structure (Structure): Pymatgen bulk structure with magnetic moments and ordering.
        deformations              : List of scaling factors that tunes the cell volume.
        magnetic_ordering         : Depeding if the magnetic configuration is ["NM", "AFM", "FM"].
        vasp_input_set            : Alternative DFT method.
        vasp_cmd                  : Environment variable for vasp execution.
        db_file                   : To connect to the database.

    Return:
        Workflow, which consist in optimization + Deformation + EOS_Fit
    """
    fws, parents = [], []

    # Bulk-optimization
    vasp_opt = MOSurfaceSet(bulk_structure, user_incar_settings={"ISIF": 3}, bulk=True)

    # linspace deformations
    if not deformations:
        deformations = [
            (np.identity(3) * (1 + x)).tolist() for x in np.linspace(-0.157, 0.157, 6)
        ]

    # Bulk structure optimization
    name_bulk = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_bulk_optimization"
    fws.append(
        Bulk_FW(
            bulk_structure,
            name=name_bulk,
            vasp_input_set=vasp_opt,
            vasp_cmd=vasp_cmd,
            db_file=db_file,
        )
    )
    parents = fws[0]

    # Deformations
    vasp_static = MOSurfaceSet(
        bulk_structure, user_incar_settings={"NSW": 0}, bulk=True
    )
    deformations = [Deformation(defo_mat) for defo_mat in deformations]
    for n, deformation in enumerate(deformations):
        name_deformation = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_deformation_{n}"
        fw = TransmuterFW(
            name=name_deformation,
            structure=bulk_structure,
            transformations=["DeformStructureTransformation"],
            transformation_params=[{"deformation": deformation.tolist()}],
            vasp_input_set=vasp_static,
            parents=parents,
            vasp_cmd=vasp_cmd,
            db_file=db_file,
        )
        fws.append(fw)

    # Fit EOS task
    parents = fws[1:]
    name_fit_eos = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_eos_fitting_analysis"
    fw_analysis = Firework(
        FitEquationOfStateFW(magnetic_ordering=magnetic_ordering, db_file=db_file),
        name=name_fit_eos,
        parents=parents,
    )
    fws.append(fw_analysis)

    # Create workflow
    wf_eos = Workflow(fws)
    wf_eos.name = (
        f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering}_EOS_WF"
    )
    return wf_eos
