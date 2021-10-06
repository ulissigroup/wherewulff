from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
    Ordering,
)

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.core.periodic_table import Element

from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
)

from fireworks import LaunchPad
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.dft_settings.settings import (
    set_bulk_magmoms,
)
from CatFlows.workflows.eos import EOS_WF

# from CatFlows.workflows.surface_energy import SurfaceEnergy_WF
# from CatFlows.workflows.wulff_shape import WulffShape_WF
# from CatFlows.workflows.slab_ads import SlabAds_WF
# from CatFlows.adsorption.adsorbate_configs import OH_Ox_list


# Bulk structure workflow method
class BulkFlows:
    """
    BulkFlow is a general method to automatize DFT workflows to find the Equilibrium Bulk
    Structure with the right magnetic moments and Ordering.

    Args:
        bulk_structure
        conventional_standard

    Returns:
        The launchpad ready for execution!
    """

    def __init__(
        self,
        bulk_structure,
        conventional_standard=True,
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
    ):

        # Bulk structure
        self.bulk_structure = self._read_cif_file(bulk_structure)
        self.original_bulk_structure = self.bulk_structure.copy()
        # Convetional standard unit cell
        if conventional_standard:
            self.bulk_structure = self._get_conventional_standard()
        # Decorate with oxidations states
        self.bulk_structure = self._get_oxidation_states()
        # Decorate the bulk structure with sites properties
        self.bulk_structure = self._get_wyckoffs_positions()
        # Get magmoms for metals
        self.magmoms_dict = self._get_metals_magmoms()
        # Get magnetic orderings
        self.magnetic_orderings_dict = self._get_magnetic_orderings()
        # Get bulk structures dict for NM, AFM, FM
        self.bulk_structures_dict = self._get_all_bulk_magnetic_configurations()

        # VASP_CMD and DB_FILE
        self.vasp_cmd = vasp_cmd
        self.db_file = db_file

        # Workflow
        self.workflows_list = self._get_all_wfs()

    def _read_cif_file(self, bulk_structure, primitive=False):
        """Parse CIF file with PMG"""
        struct = CifParser(bulk_structure).get_structures(primitive=primitive)[0]
        return struct

    def _get_oxidation_states(self):
        """Decorates bulk with oxidation states"""
        oxid_transformer = AutoOxiStateDecorationTransformation()
        struct_new = oxid_transformer.apply_transformation(self.bulk_structure)
        return struct_new

    def _get_conventional_standard(self):
        """Convert Bulk structure to conventional standard"""
        SGA = SpacegroupAnalyzer(self.bulk_structure)
        bulk_structure = SGA.get_conventional_standard_structure()
        return bulk_structure

    def _get_wyckoffs_positions(self):
        """Decorates the bulk structure with wyckoff positions"""
        bulk_structure = self.bulk_structure.copy()
        SGA = SpacegroupAnalyzer(bulk_structure)
        bulk_structure.add_site_property(
            "bulk_wyckoff", SGA.get_symmetry_dataset()["wyckoffs"]
        )
        bulk_structure.add_site_property(
            "bulk_equivalent", SGA.get_symmetry_dataset()["equivalent_atoms"].tolist()
        )
        return bulk_structure

    def _get_metals_magmoms(self):
        """Returns dict with metal symbol and magmoms assigned"""
        bulk_structure = set_bulk_magmoms(self.bulk_structure)
        metals_symb = [
            metal
            for metal in self.original_bulk_structure.species
            if Element(metal).is_metal
        ]
        magmoms_list = bulk_structure.site_properties["magmom"]
        magmoms_dict = {}
        for metal, magmom in zip(metals_symb, magmoms_list):
            magmoms_dict.update({str(metal): magmom})
        return magmoms_dict

    def _get_magnetic_orderings(self):
        """Returns a dict with AFM and FM magnetic structures orderings"""
        enumerator = MagneticStructureEnumerator(
            self.bulk_structure,
            default_magmoms=self.magmoms_dict,
            automatic=True,
            truncate_by_symmetry=True,
        )
        ordered_structures = enumerator.ordered_structures

        magnetic_orderings_dict = {}
        for ord_struct in ordered_structures:
            analyzer = CollinearMagneticStructureAnalyzer(ord_struct)
            struct_with_spin = analyzer.get_structure_with_spin()
            struct_dict = struct_with_spin.as_dict()
            if analyzer.ordering == Ordering.AFM:
                if struct_with_spin.num_sites == self.original_bulk_structure.num_sites:
                    afm_magmom = [
                        x["species"][0]["properties"]["spin"]
                        for x in struct_dict["sites"]
                    ]
                    magnetic_orderings_dict.update({"AFM": afm_magmom})
            elif analyzer.ordering == Ordering.FM:
                if struct_with_spin.num_sites == self.original_bulk_structure.num_sites:
                    fm_magmom = [
                        x["species"][0]["properties"]["spin"]
                        for x in struct_dict["sites"]
                    ]
                    magnetic_orderings_dict.update({"FM": fm_magmom})
        return magnetic_orderings_dict

    def _get_all_bulk_magnetic_configurations(self):
        """Decorates the original bulk structure with NM, AFM and FM"""
        bulk_structure = self.original_bulk_structure.copy()
        magnetic_orderings = self.magnetic_orderings_dict
        # Add AFM and FM
        bulk_structures_dict = {}
        for k, v in magnetic_orderings.items():
            bulk_new = bulk_structure.copy()
            bulk_new.add_site_property("magmom", v)
            bulk_structures_dict.update({k: bulk_new.as_dict()})
        # Add NM
        bulk_structures_dict.update({"NM": bulk_structure.as_dict()})
        return bulk_structures_dict

    def _get_all_wfs(self):
        """Returns the list of workflows to be launched"""
        # wfs for NM + AFM + FM
        wfs = []
        for mag_ord, bulk_struct in self.bulk_structures_dict:
            eos_wf = EOS_WF(
                bulk_struct,
                magnetic_ordering=mag_ord,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
            )
            wfs.append(eos_wf)
        return wfs

    def _get_parents(self, workflow_list):
        """Returns an unpacked list of parents from a set of wfs"""
        wf_fws = [wf.fws for wf in workflow_list]
        fws = [fw for wf in wf_fws for fw in wf]
        return fws

    def submit(self, hostname, db_name, port, username, password, reset=False):
        """Submit Full Workflow to Launchpad!"""
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

        return launchpad
