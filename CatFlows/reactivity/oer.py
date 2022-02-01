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
        metal_site   (string: Ir)     : User selected metal_site composition when the material is bi-metallic
        adsorbates   (Dict)           : Is a dict of the well-known OER-WNA adsorbates (OH, Ox, OOH)
        random_state (default: 42)    : This method should choose the active site automatically and (pseudo-randomly)

    Return:
        A dictionary of intermediates e.g. {"reference", "OH_0", "OH_1",...,"OOH_up_0",..., "OOH_down_0",...,}
    """

    def __init__(self, slab, slab_orig, bulk_like_sites, metal_site="", adsorbates=oer_adsorbates_dict, random_state=42):
        self.slab = slab
        self.slab_orig = slab_orig
        self.bulk_like_sites = bulk_like_sites
        self.metal_site = metal_site
        self.adsorbates = adsorbates
        self.random_state = random_state

        # Inspect slab site properties to determine termination (OH/Ox)
        (
            self.surface_coverage,
            self.ads_species,
            self.ads_indices,
            self.termination_info,
        ) = self._get_surface_termination()

        # Cache all the idx
        self.all_ads_indices = self.ads_indices

        # Select active site composition
        active_sites_dict = self._group_ads_sites_by_metal()
        assert self.metal_site in active_sites_dict.keys(), f"There is no available {self.metal_site} on the surface"
        self.ads_indices = active_sites_dict[self.metal_site]

        # Generate slab reference to place the adsorbates
        self.ref_slab, self.reactive_idx = self._get_reference_slab()

        # Shifted bulk_like_sites
        self.bulk_like_sites, self.bulk_like_dict = self._get_shifted_bulk_like_sites()

        # Selected site
        self.selected_site = self.bulk_like_dict[self.reactive_idx]

        # Print
        print(self.ads_indices, self.reactive_idx, self.bulk_like_dict)


#        self.mxidegen = self._mxidegen()
#        self.bulk_like_sites, _ = self.mxidegen.get_bulk_like_adsites()

        # Select bulk-like site nearest to "removed" oxo site
#        if len(self.bulk_like_sites) > 1:
#            self.selected_site = self._find_nearest_bulk_like_site(
#                reactive_idx=self.reactive_idx
#            )
#        else:
#            self.selected_site = self.bulk_like_sites

        # np seed
        np.random.seed(self.random_state)

    def _get_surface_termination(self):
        """Helper function to get whether the surface is OH or Ox terminated"""
        termination_info = [
            [idx, site.specie, site.frac_coords]
            for idx, site in enumerate(self.slab.sites)
            if "surface_properties" in site.properties and site.properties["surface_properties"] == "adsorbate"
        ]

        # Filter sites information
        ads_species = [site[1] for site in termination_info]
        ads_indices = [site[0] for site in termination_info]

        # OH or Ox coverage
        surface_coverage = ["oxo" if Element("H") not in ads_species else "oh"]
        return surface_coverage, ads_species, ads_indices, termination_info

    def _find_nearest_bulk_like_site(self, bulk_like_sites, reactive_idx):
        """Find reactive site by min distance between bulk-like and selected reactive site"""
        ox_site = [site for idx, site in enumerate(self.slab) if idx == reactive_idx][0]

        min_dist = np.inf
        for bulk_like_site in bulk_like_sites:
            dist = np.linalg.norm(bulk_like_site - ox_site.coords)
            if dist <= min_dist:
                min_dist = dist
                nn_site = bulk_like_site
        return nn_site

    def _find_nearest_hydrogen(self, site_idx, search_list):
        """Depending on how the surface atoms are sorted we need to find the nearest H"""
        fixed_site = [site for idx, site in enumerate(self.slab) if idx == site_idx][0]

        min_dist = np.inf
        for site in search_list:
            dist = np.linalg.norm(fixed_site.frac_coords - site[2])
            if dist <= min_dist:
                min_dist = dist
                nn_site = site
        return nn_site[0]

    def _find_nearest_metal(self, reactive_idx):
        """Find reactive site by min distance between any metal and oxygen"""
        reactive_site = [site for idx, site in enumerate(self.slab) if idx == reactive_idx][0]

        min_dist = np.inf
        for site in self.slab:
            if Element(site.specie).is_metal:
                dist = site.distance(reactive_site)
                if dist <= min_dist:
                    min_dist = dist
                    closest_metal = site
        return closest_metal

    def _group_ads_sites_by_metal(self):
        """Groups self.ads_indices by metal"""
        sites_by_metal = {}
        for ads_idx in self.ads_indices:
            close_site = self._find_nearest_metal(ads_idx)
            if str(close_site.specie) not in sites_by_metal.keys():
                sites_by_metal.update({str(close_site.specie): [ads_idx]})
            elif str(close_site.specie) in sites_by_metal.keys():
                sites_by_metal[str(close_site.specie)].append(ads_idx)
        return sites_by_metal

    def _get_reference_slab(self):
        """Random selects a termination adsorbate and clean its"""

        if self.surface_coverage[0] == "oxo":
            ref_slab = self.slab.copy()
            reactive_site = np.random.choice(self.ads_indices)
            reactive_site = 101
            ref_slab.remove_sites(indices=[reactive_site])

            return ref_slab, reactive_site

        elif self.surface_coverage[0] == "oh":
            ref_slab = self.slab.copy()
            ads_indices_oxygen = [
                site[0] for site in self.termination_info if site[1] == Element("O")
            ]
            ads_indices_hyd = [
                site for site in self.termination_info if site[1] == Element("H")
            ]
            reactive_site_oxygen = np.random.choice(ads_indices_oxygen)
            hyd_site = self._find_nearest_hydrogen(
                reactive_site_oxygen, ads_indices_hyd
            )
            reactive_site = [reactive_site_oxygen, hyd_site]
            ref_slab.remove_sites(indices=reactive_site)

            return ref_slab, reactive_site_oxygen

    def _mxidegen(self, repeat=[1, 1, 1], verbose=False):
        """Returns the MXide Method for the ref_slab"""
        mxidegen = MXideAdsorbateGenerator(
            self.ref_slab, # 110 -> 3O* + 1 (*) and slab_orig 4(*)
            repeat=repeat,
            verbose=verbose,
            positions=["MX_adsites"],
            relax_tol=0.025,
        )
        return mxidegen

    def _get_shifted_bulk_like_sites(self, repeat=[1,1,1], verbose=False):
        """Get Perturbed bulk-like sites"""
        # Mxide on pristine slab
        #mxidegen = MXideAdsorbateGenerator(
        #    self.slab_orig,
        #    repeat=repeat,
        #    verbose=verbose,
        #    positions=['MX_adsites'],
        #    relax_tol=0.025
        #)

        # Pristine bulk_like sites
        #bulk_like, _ = mxidegen.get_bulk_like_adsites()
        #bondlength, X = mxidegen.bondlength, mxidegen.X

        # Create the clean slab
        #clean_slab = self._get_clean_slab()

        # Perturb pristine bulk_like sites
        bulk_like_shifted = self._bulk_like_adsites_perturbation_oxygens(self.slab_orig, self.slab)

        # Sort the bulk_like_sites with ads_idx
        bulk_like_dict = {} # {idx: [x,y,z]}
#        min_dist = np.inf
#        for bulk_like_site in bulk_like_shifted:
#            for idx, site in enumerate(self.slab):
#                if site.specie == Element(X) and site.coords[2] > self.slab.center_of_mass[2]:
#                    dist = np.linalg.norm(bulk_like_site - site.coords)
#                    if dist <= min_dist:
#                        bulk_like_dict.update({idx: bulk_like_site})

        for ads_idx in self.all_ads_indices:
            nn_site = self._find_nearest_bulk_like_site(bulk_like_shifted, ads_idx)
            bulk_like_dict.update({ads_idx: nn_site})

        return bulk_like_shifted, bulk_like_dict

    def _bulk_like_adsites_perturbation(self, slab_ref, slab, bulk_like_sites, bondlength, X):
        """Let's perturb bulk_like_sites with delta (xyz)"""
        slab_ref_coords = slab_ref.cart_coords
        slab_coords = slab.cart_coords

        delta_coords = slab_coords - slab_ref_coords

        metal_idx = []
        for bulk_like_site in bulk_like_sites:
            for idx, site in enumerate(slab_ref):
                if site.specie != Element(X) and site.coords[2] > slab_ref.center_of_mass[2]:
                    dist = np.linalg.norm(bulk_like_site - site.coords)
                    if dist < bondlength:
                        metal_idx.append(idx)

        bulk_like_deltas = [delta_coords[i] for i in metal_idx]
        return [n+m for n, m in zip(bulk_like_sites, bulk_like_deltas)]

    def _bulk_like_adsite_perturbation_oxygens(self, slab_ref, slab):
        """peturbation on oxygens"""
        slab_ref_coords = slab_ref.cart_coords # input
        slab_coords = slab.cart_coords # output

        delta_coords = slab_coords - slab_ref_coords

        ox_idx = []
        min_dist = np.inf
        for bulk_like_site in self.bulk_like_sites:
            for idx, site in enumerate(slab_ref):
                if site.specie == Element("O") and site.coords[2] > slab_ref.center_of_mass[2]:
                    dist = np.linalg.norm(bulk_like_site - site.coords)
                    if dist <= min_dist:
                        min_dist = dist
                        ox_idx.append(idx)

        bulk_like_deltas = [delta_coords[i] for i in ox_idx]
        return [n+m for n, m in zip(bulk_like_site, bulk_like_deltas)]


    def _get_clean_slab(self):
        """Remove all the adsorbates"""
        clean_slab = self.slab.copy()
        clean_slab.remove_sites(indices=self.all_ads_indices)
        return clean_slab

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

    def _add_adsorbates(self, adslab, ads_coords, molecule, z_offset=[0, 0, 0.15]):
        """Add molecule in the open coordination site"""
        translated_molecule = molecule.copy()
        for ads_site in ads_coords:
            for mol_site in translated_molecule:
                new_coord = ads_site + (mol_site.coords - z_offset)
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
