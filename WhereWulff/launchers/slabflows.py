from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from atomate.vasp.config import DB_FILE, VASP_CMD
from fireworks import LaunchPad, Workflow
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.surface import (SlabGenerator,
                                   get_symmetrically_distinct_miller_indices)
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import \
    AutoOxiStateDecorationTransformation
from WhereWulff.adsorption.adsorbate_configs import OH_Ox_list
from WhereWulff.dft_settings.settings import (SelectiveDynamics,
                                              set_bulk_magmoms)
from WhereWulff.workflows.oer import OER_WF
from WhereWulff.workflows.slab_ads import SlabAds_WF
from WhereWulff.workflows.surface_energy import SurfaceEnergy_WF
from WhereWulff.workflows.wulff_shape import WulffShape_WF


# Surface Workflow method
class SlabFlows:
    """
    SlabFlows is a general method to automatize DFT Workflows for Surface Chemistry and Catalysis.

    Args:
        bulk_structure                          : CIF file path.
        conventional_standard (default: True)   : To select if bulk structure should be conventional standard.
        add_magmoms           (default: True)   : Decorates bulk structure with MAGMOM based on Crystal field Theory.
        include_bulk_opt      (default: True)   : To select if oriented bulk should be optimized (required for surface energy analysis).
        max_index             (default: 1)      : Maximum number for (h,k,l) miller indexes.
        symmetrize            (default: True)   : To enforce that top/bottom layers are symmetrized while slicing the slab model.
        slab_repeat           (default: [2,2,1]): Slab model supercell in the xy plane.
        selective_dynamics    (default: False)  : Contraint bottom-half of the slab model.
        wulff_analysis        (default: True)   : Add Wulff shape Analysis in the workflow (To prioritize surfaces).
        exclude_hkl           (default: list)   : List of tupple miller indexes [(h, k, l), (h', k', l')] to not compute.
        stop_at_wulff_an      (default: False)  : Stop workflow at Wulff Shape level. (avoid pbx and reactivity).
        adsorbates_list       (default: List)   : List of adsorbates as Molecule PMG objects (OH/Ox)
        applied_potential     (default: 1.60)   : Applied potential to determine the most stable termination at given voltage.
        applied_pH            (default: 0.0)    : Applied pH to determine the most stable termination at give pH.
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
        slab_repeat=[2, 2, 1],
        symmetrize=True,
        selective_dynamics=False,
        exclude_hkl=None,
        stop_at_wulff_analysis=False,
        adsorbates=OH_Ox_list,
        applied_potential=1.60,
        applied_pH=0,
        metal_site="",
        vasp_input_set=None,
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
        run_fake=False,
        streamline=False,
        checkpoint_path=None,
    ):

        self.run_fake = run_fake
        # Bulk structure
        self.bulk_structure = self._read_cif_file(bulk_structure)
        if conventional_standard:
            self.bulk_structure = self._get_conventional_standard()
        if add_magmoms:
            self.bulk_structure = set_bulk_magmoms(self.bulk_structure)

        # Slab modeling parameters
        self.include_bulk_opt = include_bulk_opt
        self.max_index = max_index
        self.symmetrize = symmetrize
        self.slab_repeat = slab_repeat
        self.selective_dynamics = selective_dynamics
        self.stop_at_wulff_analysis = stop_at_wulff_analysis
        self.exclude_hkl = exclude_hkl

        # DFT method and vasp_cmd and db_file
        self.vasp_input_set = vasp_input_set
        self.vasp_cmd = vasp_cmd
        self.db_file = db_file

        # Reactivite site
        self.metal_site = metal_site

        # General info
        self.bulk_formula = self._get_bulk_formula()
        self.miller_indices = self._get_miller_indices()
        self.slab_structures = self._get_slab_structures()
        self.workflows_list = self._get_all_wfs()
        self.adsorbates = adsorbates

        # PBX conditions
        self.applied_potential = applied_potential
        self.applied_pH = applied_pH

        # ML used to streamline
        self.streamline = streamline
        self.checkpoint_path = checkpoint_path
        if self.streamline and self.checkpoint_path is None:
            raise ValueError(
                "If wish to streamline need to provide checkpoint absolute path on server"
            )

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

    def _get_bulk_formula(self):
        """Returns Bulk formula"""
        bulk_formula = self.bulk_structure.composition.reduced_formula
        return bulk_formula

    def _get_miller_indices(self):
        """Returns a list of Crystallographic orientations (hkl)"""
        miller_indices = get_symmetrically_distinct_miller_indices(
            self.bulk_structure, max_index=self.max_index
        )
        if self.exclude_hkl:
            miller_indices = set(miller_indices) - set(self.exclude_hkl)
        return list(miller_indices)

    def _get_miller_vector(self, slab):
        """Returns the unit vector aligned with the miller index."""
        mvec = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
        return mvec / np.linalg.norm(mvec)

    def _count_surface_metals(self, slab):
        """Check whether the metal_site is on top of the surface for reactivity"""
        spg = SpacegroupAnalyzer(slab.oriented_unit_cell)
        ucell = spg.get_symmetrized_structure()
        v = VoronoiNN()
        unique_indices = [equ[0] for equ in ucell.equivalent_indices]

        # Check oriented cell atoms coordination
        cn_dict = {}
        for i in unique_indices:
            el = ucell[i].species_string
            if el not in cn_dict.keys():
                cn_dict[el] = []
            cn = v.get_cn(ucell, i, use_weights=True)
            cn = float("%.5f" % (round(cn, 5)))
            if cn not in cn_dict[el]:
                cn_dict[el].append(cn)

        # Check if metal_site in top layer
        active_sites = []
        for i, site in enumerate(slab):
            if site.frac_coords[2] > slab.center_of_mass[2]:
                if self.metal_site in site.species_string:
                    cn = float("%.5f" % (round(v.get_cn(slab, i, use_weights=True), 5)))
                    if cn < min(cn_dict[site.species_string]):
                        active_sites.append(site)

        # Min c parameter reference to bottom layer
        bottom_c = min([site.c for site in slab])

        return sum([site.c - bottom_c for site in active_sites])

    def _get_slab_structures(self, ftol=0.01):
        """Returns a list of slab structures"""
        slab_list = []
        for mi_index in self.miller_indices:
            slab_gen = SlabGenerator(
                self.bulk_structure,
                miller_index=mi_index,
                min_slab_size=3,  # fixed for test purposes
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

            slab_candidates = []
            for slab in all_slabs:
                slab_formula = slab.composition.reduced_formula
                if (
                    not slab.is_polar()
                    and slab.is_symmetric()
                    and slab_formula == self.bulk_formula  # Only for test purposes
                ):
                    slab.make_supercell(self.slab_repeat)
                    slab_candidates.append(slab)
            # This is new!
            if len(slab_candidates) >= 1:
                count_metal = 0
                for slab_cand in slab_candidates:
                    count = self._count_surface_metals(slab_cand)
                    if count > count_metal:
                        count_metal = count
                        slab_list.append(slab_cand)
            print(mi_index)
            breakpoint()
        return slab_list

    def _get_all_wfs(self):
        """Returns the list of workflows to be launched"""
        # wfs for oriented bulk + slab model
        wfs = []
        for slab in self.slab_structures:
            if self.selective_dynamics:
                slab = SelectiveDynamics.center_of_mass(slab)
            slab_wf = SurfaceEnergy_WF(
                slab,
                include_bulk_opt=self.include_bulk_opt,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
                run_fake=self.run_fake,
            )
            wfs.append(slab_wf)
        return wfs

    def _get_wulff_analysis(self, parents=None):
        """Returns Wulff Shape analysis"""
        wulff_wf, parents_fws = WulffShape_WF(
            self.bulk_structure,
            parents=parents,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
        )
        return wulff_wf, parents_fws

    def _get_ads_slab_wfs(self, parents=None):
        """Returns all the Ads_slabs fireworks"""
        ads_slab_wfs, parents_fws = SlabAds_WF(
            self.bulk_structure,
            self.adsorbates,
            parents=parents,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
            run_fake=self.run_fake,
            metal_site=self.metal_site,
            applied_potential=self.applied_potential,
            applied_pH=self.applied_pH,
            streamline=self.streamline,
            checkpoint_path=self.checkpoint_path,
        )
        return ads_slab_wfs, parents_fws

    def _get_oer_reactivity(self, parents=None):
        """Returns all the OER ads_slab fireworks"""
        oer_fws = []
        for hkl in self.miller_indices:
            miller_index = "".join(list(map(str, hkl)))
            oer_fw = OER_WF(
                self.bulk_structure,
                miller_index,
                metal_site=self.metal_site,
                applied_potential=self.applied_potential,
                applied_pH=self.applied_pH,
                parents=parents,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
                run_fake=self.run_fake,
            )
            oer_fws.append(oer_fw)

        # convert fws list into wf
        wf_name = f"{self.bulk_structure.composition.reduced_formula}-{miller_index} OER Single Site WNA"
        oer_wf = self._convert_to_workflow(oer_fws, name=wf_name, parents=parents)
        return oer_wf

    def _get_oer_reactivity_new(self, parents=None):
        """New OER but remove inner links"""
        miller_index_list = [
            "".join(list(map(str, hkl))) for hkl in self.miller_indices
        ]
        oer_wf = OER_WF_new(
            bulk_structure=self.bulk_structure,
            miller_index_list=miller_index_list,
            parents=parents,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
        )
        return oer_wf

    def _get_parents(self, workflow_list):
        """Returns an unpacked list of parents from a set of wfs"""
        wf_fws = [wf.fws for wf in workflow_list]
        fws = [fw for wf in wf_fws for fw in wf]
        return fws

    def _convert_to_workflow(self, fws_list, name="", parents=None):
        """Helper function that converts list of fws into a workflow"""
        if parents is not None:
            fws_list.extend(parents)
        wf = Workflow(fws_list, name=name)
        return wf

    def submit(self, hostname, db_name, port, username, password, reset=False):
        """Submit Full Workflow to Launchpad !"""
        launchpad = (
            LaunchPad(
                host=hostname,
                name=db_name,
                port=port,
                username=username,
                password=password,
            )
            if hostname
            else LaunchPad()
        )
        if reset:
            launchpad.reset("", require_password=False)

        parents_list = self._get_parents(self.workflows_list)

        # Wulff shape analysis
        if self.stop_at_wulff_analysis:
            wulff_wf = self._get_wulff_analysis(parents=parents_list)
            launchpad.add_wf(wulff_wf)

        else:
            # Add OER reactivity
            if len(self.miller_indices) > 1:
                wulff_wf, wulff_parents = self._get_wulff_analysis(parents=parents_list)

                # Surface Pourbaix Diagram (OH/Ox)
                ads_slab_wf, ads_slab_fws = self._get_ads_slab_wfs(
                    parents=wulff_parents
                )
            else:  # case where there is only one surface orientation
                # skip the Wulff
                ads_slab_wf, ads_slab_fws = self._get_ads_slab_wfs(parents=parents_list)

            # Reactivity -> OER
            # oer_wf = self._get_oer_reactivity(parents=ads_slab_fws)
            # launchpad.add_wf(oer_wf)
            #            launchpad = LaunchPad()
            launchpad.add_wf(ads_slab_wf)

            # Loop over OER
            # for oer_wf in range(len(oer_wfs)):
            #    launchpad.add_wf(oer_wfs[oer_wf])

        return launchpad

    def submit_local(self, reset=True):
        """Submit Full Workflow to Launchpad !"""
        launchpad = LaunchPad()

        if reset:
            launchpad.reset("", require_password=False)

        parents_list = self._get_parents(self.workflows_list)

        # Wulff shape analysis
        if self.stop_at_wulff_analysis:
            wulff_wf = self._get_wulff_analysis(parents=parents_list)
            launchpad.add_wf(wulff_wf)

        else:
            # wulff_wf, wulff_parents = self._get_wulff_analysis(parents=parents_list)
            # Add OER reactivity
            if len(self.miller_indices) > 1:
                wulff_wf, wulff_parents = self._get_wulff_analysis(parents=parents_list)

                # Surface Pourbaix Diagram (OH/Ox)
                ads_slab_wf, ads_slab_fws = self._get_ads_slab_wfs(
                    parents=wulff_parents
                )
            else:  # case where there is only one surface orientation
                # skip the Wulff
                ads_slab_wf, ads_slab_fws = self._get_ads_slab_wfs(parents=parents_list)

            # Ads slab into the launchpad
            launchpad.add_wf(ads_slab_wf)

        return launchpad
