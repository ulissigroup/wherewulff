from __future__ import absolute_import, division, print_function, unicode_literals

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
)

from fireworks import LaunchPad
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.dft_settings.settings import MOSurfaceSet
from CatFlows.fireworks.optimize import Slab_FW
from CatFlows.firetasks.surface_energy import SurfaceEnergyFireTask
from CatFlows.workflows.surface_energy import SurfaceEnergy_WF
from CatFlows.workflows.wulff_shape import WulffShape_WF


# CatFlows Workflow method
class CatFlows:
    """
    CatFlows is a general method to automatize DFT Workflows for Surface Chemistry and Catalysis.

    Args:
        bulk_structure                          : CIF file path.
        conventional_standard (default: True)   : To select if bulk structure should be conventional standard.
        add_magmoms           (default: True)   : Decorates bulk structure with MAGMOM based on Crystal field Theory.
        include_bulk_opt      (default: True)   : To select if oriented bulk should be optimized (required for surface energy analysis).
        max_index             (default: 1)      : Maximum number for (h,k,l) miller indexes.
        symmetrize            (default: True)   : To enforce that top/bottom layers are symmetrized while slicing the slab model.
        slab_repeat           (default: [2,2,1]): Slab model supercell in the xy plane.
        wulff_analysis        (default: True)   : Add Wulff shape Analysis in the workflow (To prioritize surfaces).
        vasp_input_set        (default: None)   : To select DFT method for surface optimizations.
        vasp_cmd                                : VASP execution command (configured in my_fworker.py file)
        db_file                                 : Directs to db.json file for mongodb database configuration.

    Returns:
        The launchpad ready for execution!
    """

    def __init__(
        self,
        bulk_structure,
        conventional_standard=True,
        add_magmoms=True,
        include_bulk_opt=True,
        max_index=1,
        symmetrize=True,
        slab_repeat=[2, 2, 1],
        wulff_analysis=True,
        vasp_input_set=None,
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
    ):

        # Bulk structure
        self.bulk_structure = self._read_cif_file(bulk_structure)
        if conventional_standard:
            self.bulk_structure = self._get_conventional_standard()
        if add_magmoms:
            self.bulk_structure = self._get_bulk_magmoms()

        # Slab modeling parameters
        self.include_bulk_opt = include_bulk_opt
        self.max_index = max_index
        self.symmetrize = symmetrize
        self.slab_repeat = slab_repeat
        self.wulff_analysis = wulff_analysis

        # DFT method and vasp_cmd and db_file
        self.vasp_input_set = vasp_input_set
        self.vasp_cmd = vasp_cmd
        self.db_file = db_file

        # General info
        self.bulk_formula = self._get_bulk_formula()
        self.miller_indices = self._get_miller_indices()
        self.slab_structures = self._get_slab_structures()
        self.workflows_list = self._get_all_wfs()

    def _read_cif_file(self, bulk_structure, primitive=False):
        """Parse CIF file with PMG"""
        struct = CifParser(bulk_structure).get_structures(primitive=primitive)[0]
        oxid_transformer = AutoOxiStateDecorationTransformation()
        struct_new = oxid_transformer.apply_transformation(struct)
        return struct_new

    def _get_conventional_standard(self):
        """Convert Bulk structure to conventional standard"""
        SGA = SpacegroupAnalyzer(self.bulk_structure)
        bulk_structure = SGA.get_conventional_standard_structure()
        return bulk_structure

    def _get_bulk_magmoms(self, tol=0.1, scale_factor=1.2):
        """Returns decorated bulk structure with magmoms"""
        struct = self.bulk_structure.copy()
        # Voronoi NN
        voronoi_nn = VoronoiNN(tol=tol)
        # SPG Analysis
        sga = SpacegroupAnalyzer(struct)
        sym_struct = sga.get_symmetrized_structure()
        # Magnetic moments
        element_magmom = {}
        for idx in sym_struct.equivalent_indices:
            site = sym_struct[idx[0]]
            if site.specie.is_transition_metal:
                cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
                cn = round(cn, 5)
                # Filter between Oh or Td coordinations
                if cn > 5.0:
                    coordination = "oct"
                else:
                    coordination = "tet"
                # Spin configuration depending on row
                if site.specie.row >= 5.0:
                    spin_config = "low"
                else:
                    spin_config = "high"  # Default
                # Magnetic moment per metal site
                magmom = site.specie.get_crystal_field_spin(
                    coordination=coordination, spin_config=spin_config
                )
                # Add to dict
                element_magmom.update(
                    {str(site.specie.name): abs(scale_factor * float(magmom))}
                )

            if site.specie.is_chalcogen:
                element_magmom.update({str(site.specie.name): 0.6})

        magmoms = [element_magmom[site.specie.name] for site in struct]

        # Decorate
        for site, magmom in zip(struct.sites, magmoms):
            site.properties["magmom"] = magmom

        return struct

    def _get_bulk_formula(self):
        """Returns Bulk formula"""
        bulk_formula = self.bulk_structure.composition.reduced_formula
        return bulk_formula

    def _get_miller_indices(self):
        """Returns a list of Crystallographic orientations (hkl)"""
        miller_indices = get_symmetrically_distinct_miller_indices(
            self.bulk_structure, max_index=self.max_index
        )
        return miller_indices

    def _get_slab_structures(self, ftol=0.01):
        """Returns a list of slab structures"""
        slab_list = []
        for mi_index in self.miller_indices:
            slab_gen = SlabGenerator(
                self.bulk_structure,
                miller_index=mi_index,
                min_slab_size=4,
                min_vacuum_size=8,
                in_unit_planes=True,
                center_slab=True,
                reorient_lattice=True,
                lll_reduce=True,
            )

            if self.symmetrize:
                all_slabs = slab_gen.get_slabs(symmetrize=self.symmetrize, ftol=ftol)

            else:
                all_slabs = slab_gen.get_slabs(symmetrize=False, ftol=ftol)

            for slab in all_slabs:
                slab_formula = slab.composition.reduced_formula
                if (
                    not slab.is_polar()
                    and slab.is_symmetric()
                    and slab_formula == self.bulk_formula
                ):
                    slab.make_supercell(self.slab_repeat)
                    slab_list.append(slab)

        return slab_list

    def _get_all_wfs(self):
        """Returns a the list of workflows to be launched"""
        # wfs for oriented bulk + slab model
        wfs = []
        for slab in self.slab_structures:
            slab_wf = SurfaceEnergy_WF(
                slab,
                include_bulk_opt=self.include_bulk_opt,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
            )
            wfs.append(slab_wf)
        return wfs

    def _get_wulff_analysis(self, parents=None):
        """Returns Wulff Shape analysis"""
        wulff_wf = WulffShape_WF(
            self.bulk_structure,
            parents=parents,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
        )
        return wulff_wf

    def _get_parents(self, workflow_list):
        """Returns an unpacked list of parents from a set of wfs"""
        wf_fws = [wf.fws for wf in workflow_list]
        fws = [fw for wf in wf_fws for fw in wf]
        return fws

    def submit(self, reset=True):
        """Submit Full Workflow to Launchpad !"""
        launchpad = LaunchPad()
        if reset:
            launchpad.reset("", require_password=False)

        parents_list = self._get_parents(self.workflows_list)

        # Wulff shape analysis
        if self.wulff_analysis:
            wulff_wf = self._get_wulff_analysis(parents=parents_list)
            launchpad.add_wf(wulff_wf)

        return launchpad
