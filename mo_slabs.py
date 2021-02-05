from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

import pymatgen
from pymatgen import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator, generate_all_slabs, get_symmetrically_distinct_miller_indices
from pymatgen.io.vasp.sets import MVLSlabSet

from fireworks import Firework, Workflow, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE

from fireworks.scripts import lpad_run

from src.mo_workflow import MOSurfaceSet, SurfaceEnergy



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


def get_slab_wf(slab, include_bulk_opt=True, vasp_input_set=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
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
    #Tag for launcher_folder
    tag = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    
    # Get Job Name
    slab_formula = slab.composition.reduced_formula
    miller_index = "".join([str(x) for x in slab.miller_index])

    #Oriented Bulk from SLAB - Same Stochiometry
    if include_bulk_opt:
        oriented_bulk = slab.oriented_unit_cell
        #oriented_bulk.make_supercell([2,2,4]) #Not needed if the N values is good.
        vis = MOSurfaceSet(oriented_bulk, bulk=True)
        fws = [OptimizeFW(name="{}_{} oriented bulk optimization".format(miller_index, tag), 
                          structure=oriented_bulk, vasp_input_set=vis, max_force_threshold=None,
                          vasp_cmd=vasp_cmd, db_file=db_file, job_type="normal")]

        parents = fws[-1]

    # VASP Input Set
    vasp_input_set = MOSurfaceSet(slab)

    # SLAB OptimizeFW
    fw = OptimizeFW(name="{}_{} slab optimization".format(miller_index, tag), 
                    structure=slab, vasp_input_set=vasp_input_set, max_force_threshold=None, 
                    vasp_cmd=vasp_cmd, db_file=db_file, parents=parents, job_type="normal")
    fws.append(fw)

    # Surface Energy Analysis
    parents = fws[1:]
    fws_analysis = Firework(SurfaceEnergy(tag=tag, db_file=db_file,
                                          slab_formula=slab_formula,
                                          miller_index=miller_index, to_db=False),
                                          name="{}_{} surface energy".format(slab_formula, miller_index),
                                          parents=parents)
    fws.append(fws_analysis)

    # Workflow
    wf_slab = Workflow(fws)
    wf_slab.name = "{}_{} slab workflow".format(slab_formula, miller_index)

    return wf_slab


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
    # Get MO slabs
    slabs = generate_mo_slabs(bulk_structure, conventional_standard=conventional_standard, max_index=max_index)

    if not slabs:
        slabs = generate_mo_slabs(bulk_structure, conventional_standard=conventional_standard, max_index=max_index, symmetrize=False) #stoichiometry

    # Workflows for oriented bulk + slab model
    wfs = []
    for slab in slabs:
        slab_wf = get_slab_wf(slab, vasp_cmd, db_file)

        wfs.append(slab_wf)

    # TODO: WulffShape Analysis

    return wfs


if __name__ == "__main__":

    # Get structure from cif file
    struct = Structure.from_file("MgO_225.cif")

    # Create the slab workflow
    slab_wf = get_wfs_mo_slabs(struct, conventional_standard=True)

    # Create the launchpad and add our workflow
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.bulk_add_wfs(slab_wf)

    #print(slab_wf[0])
    #print(slab_wf[1])

    #for n in slab_wf:
    #    print(n)

    #print(launchpad.get_wf_summary_dict([4, 5, 6]))

    #print(len(slab_wf))
    #lpad_run.check_wf()

    # Run slab workflow
    rapidfire(launchpad)

