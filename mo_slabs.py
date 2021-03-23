from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

import pymatgen
from pymatgen import Structure, Lattice
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

from src.mo_workflow import MOSurfaceSet, SurfaceEnergyFW, WulffShapeFW



def generate_mo_slabs(bulk_structure, conventional_standard=True, max_index=1, symmetrize=True):
    """
    MO slab models constructor.

    Args:
        bulk_structure (Structure): bulk structure from which to construct slabs.
        conventional_standard (default: True): If material requires conventional standard bulk cell.
        max_index (default: 1) : Only consider low miller indices for crystallographic orientations.
        symmetrize (default: True): Try to symmetrize top and bottom layers in the slab model.

    Returns:
        List of low-index slab models.  
    """
    # Get conventional_standard if requiered
    if conventional_standard:
        SGA = SpacegroupAnalyzer(bulk_structure)
        bulk_structure = SGA.get_conventional_standard_structure()

    # Bulk formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # Get Miller Indices
    mi_indices = get_symmetrically_distinct_miller_indices(bulk_structure, 
                                                           max_index=max_index)

    # SlabGen
    slab_list = []
    for mi_index in mi_indices:
        slabgen = SlabGenerator(bulk_structure,
                                miller_index=mi_index,
                                min_slab_size=4,
                                min_vacuum_size=8,
                                in_unit_planes=True,
                                center_slab=True,
                                reorient_lattice=True,
                                lll_reduce=True)

        if symmetrize:
            all_slabs = slabgen.get_slabs(symmetrize=symmetrize, ftol=0.01)

            for slab in all_slabs:
                slab_formula = slab.composition.reduced_formula
                miller_index = "".join([str(x) for x in slab.miller_index])
                if not slab.is_polar() and slab.is_symmetric() and slab_formula == bulk_formula:
                    slab.make_supercell([2,2,1])
                    slab_list.append(slab)
        else:
            all_slabs = slabgen.get_slabs(symmetrize=False, ftol=0.01)

            for slab in all_slabs:
                slab_formula = slab.composition.reduced_formula
                miller_index = "".join([str(x) for x in slab.miller_index])
                if not slab.is_polar() and slab_formula == bulk_formula:
                    slab.make_supercell([2,2,1])
                    slab_list.append(slab)

    return slab_list

def Slab_FW(slab, name="", parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE, add_slab_metadata=True):
    """
    Function to generate a slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab (Slab Object): Slab corresponding to the slab to be calculated.
        name (string): name of firework
        parents (): parent FWs
        vasp_cmd: vasp_comand
        db_file: path to the dabase file
        add_slab_metadata (bool): whether to add slab metada to task doc

    Returns:
        Firework correspoding to slab calculation.
    """

    # DFT Method
    vasp_input_set = MOSurfaceSet(slab, bulk=False)

    # FW
    fw = OptimizeFW(name=name, structure=slab, max_force_threshold=None,
                    vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                    db_file=db_file, parents=parents, job_type="normal")

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[-1]["additional_fields"].update(
            {"slab": slab, "parent_structure": slab.oriented_unit_cell,
             "parent_structure_metadata": parent_structure_metadata})

    return fw

def Slab_WF(slab, include_bulk_opt=True, vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
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

def get_wfs_mo_slabs(bulk_structure, conventional_standard=True, max_index=1, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Collects all workflows to every slab model generated from bulk_structure

    Args:
        bulk_structure (Structure): bulk structure to build slab models from it.
        conventional_standard (default: True): If material requires conventional standard bulk cell.
        max_index (default: 1): Only lower miller indexes (by now).
        vasp_cmd: vasp executable.
        db_file: database file.

    Returns:
        List of workflows.
    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # Get MO slabs
    slabs = generate_mo_slabs(bulk_structure, 
                              conventional_standard=conventional_standard, 
                              max_index=max_index)

    if not slabs:
        slabs = generate_mo_slabs(bulk_structure, 
                                  conventional_standard=conventional_standard, 
                                  max_index=max_index, symmetrize=False) #stoichiometry

    # Workflows for oriented bulk + slab model
    wfs = []
    for slab in slabs:
        slab_wf = Slab_WF(slab, include_bulk_opt=True, vasp_cmd=VASP_CMD, db_file=DB_FILE)

        wfs.append(slab_wf)

    return wfs

def WulffShape_WF(bulk_structure, vasp_cmd=VASP_CMD, db_file=DB_FILE):
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
    wulff_fw = [Firework(WulffShapeFW(bulk_structure=bulk_structure,
                                      db_file=db_file), name="{} wulff shape Task".format(bulk_formula))]

    wulff_wf = Workflow(wulff_fw, name="{} wulff shape analysis".format(bulk_formula))

    return wulff_wf



# Execute!
if __name__ == "__main__":

    # Get structure from cif file
    struct = Structure.from_file("MgO_225.cif")

    # Create the slab workflow
    slab_wfs = get_wfs_mo_slabs(struct, conventional_standard=True)

    # Create the launchpad and add our workflow
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.bulk_add_wfs(slab_wfs)

    # Run slab workflow
    rapidfire(launchpad)

    # Wulff
    wulff_wfs = WulffShape_WF(struct)
    launchpad.add_wf(wulff_wfs)

    # Run Wulff workflow
    rapidfire(launchpad)

#End