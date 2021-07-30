from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
from workflows.mo_slabs import WulffShape_WF

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

from dft_settings.settings import MOSurfaceSet
from firetasks.surface_energy import SurfaceEnergyFW
from fireworks.optimize import Slab_FW
from analysis.wulff_shape import WulffShapeFW
from surface_energy import SurfaceEnergy_WF


# General workflow method
class DaaamnYuri():
    """ General Workflow class """
    def __init__(self, bulk_structure, 
                       conventional_standard=True,
                       include_bulk_opt=True, 
                       max_index=1, 
                       symmetrize=True, 
                       vasp_input_set=None,
                       vasp_cmd=VASP_CMD,
                       db_file=DB_FILE):

        self.bulk_structure = bulk_structure
        if conventional_standard:
            self.bulk_structure = self._get_conventional_standard()
        self.include_bulk_opt = include_bulk_opt # ToDo: makes sense?
        self.max_index = max_index
        self.symmetrize = symmetrize
        self.vasp_input_set = vasp_input_set

        self.vasp_cmd = vasp_cmd
        self.db_file = db_file

        self.bulk_formula = self._get_bulk_formula()
        self.miller_indices = self._get_miller_indices()
        self.slab_structures = self._get_slab_structures()
        self.workflows_list = self._get_all_wfs()

    def _get_conventional_standard(self):
        """ Convert Bulk structure to conventional standard """
        SGA = SpacegroupAnalyzer(self.bulk_structure)
        bulk_structure = SGA.get_conventional_standard_structure()
        return bulk_structure

    def _get_bulk_formula(self):
        """ Returns Bulk formula """
        bulk_formula = self.bulk_structure.composition.reduced_formula
        return bulk_formula

    def _get_miller_indices(self):
        """ Returns a list of Crystallographic orientations (hkl) """
        miller_indices = get_symmetrically_distinct_miller_indices(self.bulk_structure,
                                                                   max_index=self.max_index)
        return miller_indices

    def _get_slab_structures(self, repeat=[2,2,1], ftol=0.01):
        """ Returns a list of slab structures """
        slab_list = []
        for mi_index in self.miller_indices:
            slab_gen = SlabGenerator(self.bulk_structure,
                                     miller_index = mi_index,
                                     min_slab_size = 4,
                                     min_vacuum_size = 8,
                                     in_unit_planes = True,
                                     center_slab = True,
                                     reorient_lattice=True,
                                     lll_reduce=True)
            
            if self.symmetrize:
                all_slabs = slab_gen.get_slabs(symmetrize=self.symmetrize, ftol=ftol)

            else:
                alls_slabs = slab_gen.get_slabs(symmetrize=False, ftol=ftol)

            for slab in all_slabs:
                slab_formula = slab.composition.reduced_formula
                miller_index = "".join([str(x) for x in slab.miller_index])
                if not slab.is_polar() and slab.is_symmetric() and slab_formula == self.bulk_formula:
                    slab.make_supercell(repeat)
                    slab_list.append(slab)

        return slab_list

    def _get_all_wfs(self):
        """ Returns a the list of workflows to be launched """
        # wfs for oriented bulk + slab model
        wfs = []
        for slab in self.slab_structures:
            slab_wf = SurfaceEnergy_WF(slab, 
                                       include_bulk_opt=self.include_bulk_opt, 
                                       vasp_cmd=self.vasp_cmd, 
                                       db_file=self.db_file)
            wfs.append(slab_wf)
        return wfs

    def _get_wulff_analysis(self):
        """ Returns Wulff Shape analysis """
        wulff_wf = WulffShape_WF(self.bulk_structure,
                                 parents=None,
                                 vasp_cmd=self.vasp_cmd,
                                 db_file=self.db_file)
        return wulff_wf

    def _get_parents(self, workflow_list):
        """ Returns an unpacked list of parents from a set of wfs """
        wf_fws = [wf.fws for wf in workflow_list]
        fws = [fw for wf in wf_fws for fw in wf]
        return fws

    def submit(self, reset=False):
        """ Submit Full Workflow to Launchpad ! """
        launchpad = LaunchPad()
        if reset:
            launchpad.reset('', requires_password=False)
        launchpad.bulk_add_wfs(self.workflows_list)

        parents_list = self._get_parents(self.workflows_list)

        # Wulff shape analysis
        if self.wulff_analysis:
            wulff_wf = self._get_wulff_analysis(parents=parents_list)
            launchpad.add_wf(wulff_wf)
        return


# Test!
if __name__ = "__main__":
    my_method = DaaamnYuri()

    my_new_firetask = my_method._get_slab_fw(slab, name)



    
