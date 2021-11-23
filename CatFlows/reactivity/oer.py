import numpy as np

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.periodic_table import Element


from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator
from CatFlows.adsorption.adsorbate_configs import oer_adsorbates_dict


class OER_SingleSite(object):
    """
    This class automatically generates the OER (single site) intermediates required to
    study the Water Nucleophilic Attack (WNA) on top of a Metal oxide surface.

    Args:
        slab         (PMG Slab object): Should be the most stable termination comming from the PBX analysis
        adsorbates   (Dict)           : Is a dict of the well-known OER-WNA adsorbates (OH, Ox, OOH)
        random_state (default: 42)    : This method should choose the active site automatically and (pseudo-randomly)

    Return:
        A dictionary of intermediates e.g. {"reference", "OH_0", "OH_1",...,"OOH_up_0",..., "OOH_down_0",...,}
    """

    def __init__(self, slab, adsorbates=oer_adsorbates_dict, random_state=42):
        self.slab = slab
        self.adsorbates = adsorbates
        self.random_state = random_state

        # methods
        (
            self.surface_coverage,
            self.ads_species,
            self.ads_indices,
            self.termination_info,
        ) = self._get_surface_termination()

        self.ref_slab = self._get_reference_slab()
        self.mxidegen = self._mxidegen()
        self.bulk_like_site, _ = self.mxidegen.get_bulk_like_adsites()

        # np seed
        np.random.seed(self.random_state)

    def _get_surface_termination(self):
        """Helper function to get whether the surface is OH or Ox terminated"""
        termination_info = [
            [idx, site.specie, site.frac_coords]
            for idx, site in enumerate(self.slab.sites)
            if site.properties["surface_properties"] == "adsorbate"
        ]

        # Filter sites information
        ads_species = [site[1] for site in termination_info]
        ads_indices = [site[0] for site in termination_info]

        # OH or Ox coverage
        surface_coverage = ["oxo" if Element("H") not in ads_species else "oh"]
        return surface_coverage, ads_species, ads_indices, termination_info

    def _get_reference_slab(self):
        """Random selects a termination adsorbate and clean its"""

        if self.surface_coverage[0] == "oxo":
            reactive_site = np.random.choice(self.ads_indices)
            ref_slab = self.slab.copy()
            ref_slab.remove_sites(indices=[reactive_site])

        elif self.surface_coverage[0] == "oh":
            ads_indices_oxygen = [
                site[0] for site in self.termination_info if site[1] == Element("O")
            ]
            reactive_site_oxygen = np.random.choice(ads_indices_oxygen, 1)
            ref_slab = self.slab.copy()
            reactive_site = [reactive_site_oxygen, reactive_site_oxygen + 1]
            ref_slab.remove_sites(indices=reactive_site)

        return ref_slab

    def _mxidegen(self, repeat=[1, 1, 1], verbose=False):
        """Returns the MXide Method for the ref_slab"""
        mxidegen = MXideAdsorbateGenerator(
            self.ref_slab, repeat=repeat, verbose=verbose
        )
        return mxidegen

    def _get_oer_intermediates(
        self, adsorbate, suffix=None, axis_rotation=[0, 0, 1], n_rotations=4
    ):
        """Returns OH/Ox/OOH intermediates"""
        # Adsorbate manipulation
        adsorbate = adsorbate.copy()
        adsorbate_label = "".join([site.species_string for site in adsorbate])
        adsorbate_angles = self._get_angles(n_rotations=n_rotations)
        adsorbate_rotations = self.mxidegen.get_transformed_molecule(
            adsorbate, axis=axis_rotation, angles_list=adsorbate_angles
        )
        # Intermediates generation
        intermediates_dict = {}
        for ads_rot_idx in range(len(adsorbate_rotations)):
            ads_slab = self.ref_slab.copy()
            ads_slab = self._add_adsorbates(
                ads_slab, self.bulk_like_site, adsorbate_rotations[ads_rot_idx]
            )
            if suffix:
                intermediates_dict.update(
                    {f"{adsorbate_label}_{suffix}_{ads_rot_idx}": ads_slab.as_dict()}
                )
            else:
                intermediates_dict.update(
                    {f"{adsorbate_label}_{ads_rot_idx}": ads_slab.as_dict()}
                )

        return intermediates_dict

    def _get_angles(self, n_rotations=4):
        """Returns the list of angles depeding on the n of rotations"""
        angles = []
        for i in range(n_rotations):
            deg = (2 * np.pi / n_rotations) * i
            angles.append(deg)
        return angles

    def _add_adsorbates(self, adslab, ads_coords, molecule, z_offset=[0,0,0.15]):
        """Add molecule in the open coordination site"""
        translated_molecule = molecule.copy()
        for ads_site in ads_coords:
            for mol_site in translated_molecule:
                new_coord = (ads_site + (mol_site.coords - z_offset))
                adslab.append(
                    mol_site.specie,
                    new_coord,
                    coords_are_cartesian=True,
                    properties=mol_site.properties,
                )
        return adslab

    def generate_oer_intermediates(self, suffix=None):
        """General method to get OER-WNA (single site) intermediantes"""
        # Get Reference slab (*)
        reference_slab = self.ref_slab.as_dict()
        reference_dict = {"reference": reference_slab}

        # Generate intermediantes depeding on Termination
        if self.surface_coverage[0] == "oxo":
            oh_intermediates = self._get_oer_intermediates(self.adsorbates["OH"])
            ooh_up = self._get_oer_intermediates(self.adsorbates["OOH_up"], suffix="up")
            ooh_down = self._get_oer_intermediates(
                self.adsorbates["OOH_down"], suffix="down"
            )
            oer_intermediates = {
                **reference_dict,
                **oh_intermediates,
                **ooh_up,
                **ooh_down,
            }
            return oer_intermediates

        elif self.surface_coverage[0] == "oh":
            ox_intermediates = self._get_oer_intermediates(
                self.adsorbates["Ox"], n_rotations=1
            )
            ooh_up = self._get_oer_intermediates(self.adsorbates["OOH_up"], suffix="up")
            ooh_down = self._get_oer_intermediates(
                self.adsorbates["OOH_down"], suffix="down"
            )
            oer_intermediates = {
                **reference_dict,
                **ox_intermediates,
                **ooh_up,
                **ooh_down,
            }
            return oer_intermediates
