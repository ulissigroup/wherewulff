from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pymatgen.analysis.elasticity.strain import Deformation

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.fireworks.core import TransmuterFW, StaticFW

from CatFlows.dft_settings.settings import MOSurfaceSet
from CatFlows.fireworks.optimize import Bulk_FW
from CatFlows.analysis.equation_of_states import FitEquationOfStateFW


def EOS_WF(bulk_structure, deformations=None, magnetic_ordering="NM", vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Equation of state workflow that handles optimization + Deformation + EOS_FIT

    Args:
        bulk_structure:
        deformations:
        magnetic_ordering:
        vasp_input_set:
        vasp_cmd:
        db_file_

    Return:
        Workflow, which consist in optimization + Deformation + EOS_Fit
    """
    fws, parents = [], []

    # linspace deformations
    if not deformations:
        deformations = [(np.identity(3) * (1 + x)).tolist() for x in np.linspace(-0.157, 0.157, 11)]

    # Bulk structure optimization
    name_bulk = f"{bulk_structure.composition.reduced_formula}_{magnetic_ordering} bulk optimization"
    fws.append(Bulk_FW(bulk_structure, name=name_bulk, vasp_cmd=vasp_cmd, db_file=db_file))
    parents = fws[0]

    # Deformations
    vasp_static = MOSurfaceSet(bulk_structure, bulk=True)
    vasp_static.incar.update({"NSW": 0})
    deformations = [Deformation(defo_mat) for defo_mat in deformations]
    for n, deformation in enumerate(deformations):
        name_deformation = ""
        fw = TransmuterFW(name=name_deformation, structure=bulk_structure,
                          transformations=["DeformStructureTransformation"],
                          transformation_params=[{"deformation": deformation.tolist()}],
                          vasp_input_set=vasp_static, parents=parents,
                          vasp_cmd=vasp_cmd, db_file=db_file)
        fws.append(fw)

    # Fit EOS task
    parents = fws[1:]
    fw_analysis = Firework(FitEquationOfStateFW(db_file=db_file),
                                                name="eos_fitting_analysis",
                                                parents=parents)
    fws.append(fw_analysis)

    # Create workflow
    wf_eos = Workflow(fws)
    wf_eos.name = f"{}_{} equation of states workflow"
    return wf_eos


    




