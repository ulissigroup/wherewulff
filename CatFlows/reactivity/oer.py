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

        self.ref_slab, self.reactive_idx = self._get_reference_slab()
        self.mxidegen = self._mxidegen()
        self.bulk_like_sites, _ = self.mxidegen.get_bulk_like_adsites()

        # Select bulk-like site nearest to "removed" oxo site
        if len(self.bulk_like_sites) > 1:
            self.selected_site = self._find_nearest_bulk_like_site(reactive_idx=self.reactive_idx)
        else:
            self.selected_site = self.bulk_like_sites

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

    def _find_nearest_bulk_like_site(self, reactive_idx):
        """ Find reactive site by min distance between bulk-like and selected reactive site """
        ox_site = [site for idx, site in enumerate(self.slab) if idx == reactive_idx][0]
        
        min_dist = np.inf
        for bulk_like_site in self.bulk_like_sites:
            dist = np.linalg.norm(bulk_like_site - ox_site.coords)
            if dist <= min_dist:
                min_dist = dist
                nn_site = bulk_like_site
        return [np.array(nn_site)]

    def _find_nearest_hydrogen(self, site_idx, search_list):
        """ Depending on how the surface atoms are sorted we need to find the nearest H """
        fixed_site = [site for idx, site in enumerate(self.slab) if idx == site_idx][0]

        min_dist = np.inf
        for site in search_list:
            dist = np.linalg.norm(fixed_site.coords - site[2])
            if dist <= min_dist:
                min_dist = dist
                nn_site = site
        return nn_site[0]

    def _get_reference_slab(self):
        """Random selects a termination adsorbate and clean its"""

        if self.surface_coverage[0] == "oxo":
            ref_slab = self.slab.copy()
            reactive_site = np.random.choice(self.ads_indices)
            ref_slab.remove_sites(indices=[reactive_site])

            return ref_slab, reactive_site

        elif self.surface_coverage[0] == "oh":
            ref_slab = self.slab.copy()
            ads_indices_oxygen = [
                site[0] for site in self.termination_info if site[1] == Element("O")
            ]
            ads_indices_hyd = [site for site in self.termination_info if site[1] == Element("H")]
            reactive_site_oxygen = np.random.choice(ads_indices_oxygen)
            hyd_site = self._find_nearest_hydrogen(reactive_site_oxygen, ads_indices_hyd)
            reactive_site = [reactive_site_oxygen, hyd_site]
            ref_slab.remove_sites(indices=reactive_site)

            return ref_slab, reactive_site_oxygen

    def _mxidegen(self, repeat=[1, 1, 1], verbose=False):
        """Returns the MXide Method for the ref_slab"""
        mxidegen = MXideAdsorbateGenerator(
            self.ref_slab, repeat=repeat, verbose=verbose, positions=["MX_adsites"], relax_tol=0.025
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
                ads_slab, self.selected_site, adsorbate_rotations[ads_rot_idx]
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
