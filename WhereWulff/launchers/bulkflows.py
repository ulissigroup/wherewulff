"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
    Ordering,
)

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
)

from fireworks import LaunchPad
from atomate.vasp.config import VASP_CMD, DB_FILE

from WhereWulff.dft_settings.settings import (
    set_bulk_magmoms,
)
from WhereWulff.workflows.eos import BulkOptimize_WF, EOS_WF
from WhereWulff.workflows.static_bulk import StaticBulk_WF
from WhereWulff.workflows.bulk_stability import StabilityBulk_WF

import warnings
warnings.filterwarnings("ignore")


# Bulk structure workflow method
class BulkFlows:
    """
    BulkFlow is a general method to automatize DFT workflows to find the Equilibrium Bulk
    Structure with the right magnetic moments and Ordering.

    Args:
        bulk_structure                          : CIF file path.
        n_deformations          (default: 21)   : Number of volume deformations for the EOS fitting.
        nm_magmom_buffer        (default: 0.6)  : VASP needs a MAGMOM buffer even if should be 0. 
        conventional_standard   (default: True) : To select if bulk structure should be conventional standard.
        vasp_cmd                                : VASP execution command (configured in my_fworkers.py file).
        db_file                                 : Directs to db.json file for mongodb database configuration.

    Returns:
        Submits the BulkFlow workflow to the launchpad and ready for execution!
    """

    def __init__(
        self,
        bulk_structure,
        n_deformations=21,
        nm_magmom_buffer=0.6,
        conventional_standard=True,
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
    ):

        # Bulk structure
        self.nm_magmom_buffer = nm_magmom_buffer
        self.bulk_structure = self._read_cif_file(bulk_structure)
        self.original_bulk_structure = self.bulk_structure.copy()
        self.n_deformations = n_deformations
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
        # self.workflows_list = self._get_all_wfs()
        # self.workflows_list = self._get_opt_wf()

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
        bulk_structure = set_bulk_magmoms(self.bulk_structure, self.nm_magmom_buffer)
        metals_symb = [
            site.species_string
            for site in self.bulk_structure
            if site.specie.element.is_metal
        ]
        magmoms_list = bulk_structure.site_properties["magmom"]
        magmoms_dict = {}
        for metal, magmom in zip(
            metals_symb, magmoms_list
        ):  # Note this results in a sublist of tuples and is OK provided the orders for both align.
            magmoms_dict.update({str(metal): magmom})
        return magmoms_dict

    def _get_magnetic_orderings(self):
        """Returns a dict with AFM and FM magnetic structures orderings"""
        magnetic_orderings_dict = {}
        try:
            enumerator = MagneticStructureEnumerator(
                self.bulk_structure,
                default_magmoms=self.magmoms_dict,
                automatic=True,
                truncate_by_symmetry=True,
            )
        except ValueError as e:
            # This is probably happening because the magmoms are zero on the
            # metal atoms. Need to catch and give a small buffer to the
            # TMO ions. Note that in this scenario, there is no need for
            # magnetic configurations.
            buffer_magmom = list(
                np.zeros(self.original_bulk_structure.num_sites, dtype=float)
                + self.nm_magmom_buffer
            )
            magnetic_orderings_dict.update({"NM": buffer_magmom})
            print(f"Encountered issue: {e}. Will give slight buffer to metal ions")
            return magnetic_orderings_dict

        ordered_structures = enumerator.ordered_structures

        for ord_struct in ordered_structures:
            analyzer = CollinearMagneticStructureAnalyzer(ord_struct)
            struct_with_spin = analyzer.get_structure_with_spin()
            struct_with_spin.sort()  # Sort so that it is aligned with original_bulk
            if analyzer.ordering == Ordering.AFM:
                if struct_with_spin.num_sites == self.original_bulk_structure.num_sites:
                    afm_magmom = [
                        float(site.specie.spin)
                        if (
                            site.specie.element.is_metal
                            and abs(float(site.specie.spin)) > 0.01
                        )
                        else self.nm_magmom_buffer
                        for site in struct_with_spin
                    ]
                    magnetic_orderings_dict.update({"AFM": afm_magmom})
            elif analyzer.ordering == Ordering.FM:
                if struct_with_spin.num_sites == self.original_bulk_structure.num_sites:
                    fm_magmom = [
                        float(site.specie.spin)
                        if (
                            site.specie.element.is_metal
                            and abs(float(site.specie.spin)) > 0.01
                        )
                        else self.nm_magmom_buffer
                        for site in struct_with_spin
                    ]
                    magnetic_orderings_dict.update({"FM": fm_magmom})

        # Adding non-magnetic
        nm_magmom = list(
            np.zeros(self.original_bulk_structure.num_sites, dtype=float)
            + self.nm_magmom_buffer
        )
        magnetic_orderings_dict.update({"NM": nm_magmom})
        return magnetic_orderings_dict

    def _get_all_bulk_magnetic_configurations(self):
        """Decorates the original bulk structure with NM, AFM and FM"""
        bulk_structure = self.original_bulk_structure.copy()
        magnetic_orderings = self.magnetic_orderings_dict
        # Add AFM, FM and NM
        bulk_structures_dict = {}
        for k, v in magnetic_orderings.items():
            bulk_new = bulk_structure.copy()
            bulk_new.add_site_property("magmom", v)
            bulk_structures_dict.update({k: bulk_new.as_dict()})
        return bulk_structures_dict

    def _get_opt_wf(self):
        """Returns bulk optimization workflow to be launched"""
        bulk_structure = Structure.from_dict(self.bulk_structures_dict["NM"])
        opt_wf, parents_fws = BulkOptimize_WF(
            bulk_structure, vasp_cmd=self.vasp_cmd, db_file=self.db_file
        )
        return opt_wf, parents_fws

    def _get_eos_wfs(self, parents=None):
        """Returns the list of workflows to be launched"""
        # wfs for NM + AFM + FM
        wfs = []
        for mag_ord, bulk_struct in self.bulk_structures_dict.items():
            bulk_struct = Structure.from_dict(bulk_struct)
            eos_wf = EOS_WF(
                bulk_struct,
                n_deformations=self.n_deformations,
                magnetic_ordering=mag_ord,
                parents=parents,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
            )
            wfs.append(eos_wf)
        return wfs

    def _get_all_wfs(self):
        """Once again"""
        eos_wf, fws_all = EOS_WF(
            self.bulk_structures_dict,
            n_deformations=self.n_deformations,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
        )
        return eos_wf, fws_all

    def _get_all_wfs_old(self):
        """Lets see"""
        wfs = []
        # Optimize
        bulk_structure = Structure.from_dict(self.bulk_structures_dict["NM"])
        opt_wf, opt_parents = BulkOptimize_WF(
            bulk_structure, vasp_cmd=self.vasp_cmd, db_file=self.db_file
        )
        # breakpoint()
        wfs.append(opt_wf)
        # OES + Fitting
        for mag_ord, bulk_struct in self.bulk_structures_dict.items():
            bulk_struct = Structure.from_dict(bulk_struct)
            eos_wf = EOS_WF(
                bulk_struct,
                magnetic_ordering=mag_ord,
                parents=opt_parents,
                vasp_cmd=self.vasp_cmd,
                db_file=self.db_file,
            )
            wfs.append(eos_wf)
        return wfs

    def _get_bulk_static_wfs(self, parents=None):
        """Returns all the BulkStatic FW"""
        bulk_static_wfs, parents_fws = StaticBulk_WF(
            self.bulk_structure,
            parents=parents,
            vasp_cmd=self.vasp_cmd,
            db_file=self.db_file,
        )
        return bulk_static_wfs, parents_fws

    def _get_stability_wfs(self, parents=None):
        """Returns all the BulkStability FW"""
        bulk_stability = StabilityBulk_WF(
            self.bulk_structure, parents=parents, db_file=self.db_file
        )
        return bulk_stability

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

        # Optimization + Deformations + EOS_FIT
        _, eos_parents = self._get_all_wfs()
        # parents_list = self._get_parents(self.workflows_list)

        # Static_FW + StabilityAnalysis
        _, static_list = self._get_bulk_static_wfs(parents=eos_parents)

        # StabilityAnalis
        bulk_stability = self._get_stability_wfs(parents=static_list)
        launchpad.add_wf(bulk_stability)

        return launchpad

    def submit_local(self, reset=True):
        """Submit Full Workflow to Launchpad!"""
        launchpad = LaunchPad()

        if reset:
            launchpad.reset("", require_password=False)

        # Optimization + Deformation + EOS_FIT
        parents_list = self._get_parents(self.workflows_list)

        # Static_FW + StabilityAnalysis
        bulk_static = self._get_bulk_static_wfs(parents=parents_list)
        launchpad.add_wf(bulk_static)

        return launchpad
