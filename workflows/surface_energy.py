from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

import pymatgen
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, get_symmetrically_distinct_miller_indices
from pymatgen.io.vasp.sets import MVLSlabSet

from fireworks import Firework, Workflow, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE


from atomate.utils.utils import get_meta_from_structure

from fireworks.scripts import lpad_run



def SurfaceEnergy_WF(slab, include_bulk_opt=True, vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Gets a workflow corresponding to a slab optimization calculation.

    Args:
        slab (Slab or Structures): Slab model to calculate.
        include_bulk_opt (default: True): Oriented bulk for surface energy calculation.
        vasp_input_set (default: MOSurfaceSet): User settings instead of default.
        vasp_cmd: vasp executable.
        db_file: database file.

    Return:
        Worflow, which consist in oriented bulk + slab model.
    """
    fws, parents = [], []
    miller_index = "".join([str(x) for x in slab.miller_index])

    # Add bulk opt firework if specified
    if include_bulk_opt:
        oriented_bulk = slab.oriented_unit_cell
        name_bulk = "{}_{} bulk optimization".format(oriented_bulk.composition.reduced_formula, miller_index)
        vis = MOSurfaceSet(oriented_bulk, bulk=True)
        fws.append(OptimizeFW(structure=oriented_bulk, name=name_bulk, vasp_input_set=vis, max_force_threshold=None,
                              vasp_cmd=vasp_cmd, db_file=db_file, job_type="normal"))
        parents = fws[0]

    # Slab model Optimization
    name_slab = "{}_{} slab optimization".format(slab.composition.reduced_formula, miller_index)
    slab_fw = Slab_FW(slab, name=name_slab, parents=parents, vasp_cmd=vasp_cmd, 
                      db_file=db_file, add_slab_metadata=True)

    fws.append(slab_fw)

    # Surface Energy Calculation
    parents = fws[1:]
    name_gamma = "{}_{} surface energy".format(slab.composition.reduced_formula, miller_index)
    gamma_hkl = Firework(SurfaceEnergyFW(slab_formula=slab.composition.reduced_formula,
                                         miller_index=miller_index, db_file=db_file, to_db=True),
                                         name=name_gamma, parents=parents)
    fws.append(gamma_hkl)

    # WF name for bulk/slab optimization
    if isinstance(slab, Slab):
        name_wf = "{}_{} slab workflow".format(slab.composition.reduced_formula, miller_index)
    else:
        name_wf = "{} slab workflow".format(slab.composition.reduced_formula)

    wf = Workflow(fws, name=name_wf)

    return wf