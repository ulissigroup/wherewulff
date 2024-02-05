import numpy as np

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.surface import Slab
from pymatgen.core.periodic_table import Element


from WhereWulff.adsorption.MXide_adsorption import MXideAdsorbateGenerator
from WhereWulff.adsorption.adsorbate_configs import oer_adsorbates_dict
from WhereWulff.utils import find_most_stable_config
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


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

    def __init__(
        self,
        slab,
        slab_orig,
        slab_clean,
        bulk_like_sites,
        metal_site="",
        adsorbates=oer_adsorbates_dict,
        random_state=42,
        # streamline=False,
        surface_coverage=None,
        checkpoint_path=None,
    ):
        self.slab = slab
        self.slab_orig = slab_orig
        self.slab_clean = slab_clean
        self.bulk_like_sites = bulk_like_sites
        self.metal_site = metal_site
        self.adsorbates = adsorbates
        self.random_state = random_state
        # self.streamline = streamline
        self.surface_coverage = [surface_coverage]
        self.checkpoint_path = checkpoint_path

        # We need to remove oxidation states
        self.slab_clean.remove_oxidation_states()
        self.slab_clean.oriented_unit_cell.remove_oxidation_states()
        # breakpoint()

        # Cache all the idx
        if not self.surface_coverage[0] == "clean":
            # Inspect slab site properties to determine termination (OH/Ox)
            (
                self.surface_coverage,
                self.ads_species,
                self.ads_indices,
                self.termination_info,
            ) = self._get_surface_termination()
            self.all_ads_indices = self.ads_indices.copy()

            # Select active site composition
            active_sites_dict = self._group_ads_sites_by_metal()
            assert (
                self.metal_site in active_sites_dict.keys()
            ), f"There is no available {self.metal_site} on the surface"
            self.ads_indices = active_sites_dict[self.metal_site]
        # TODO: Need to create all possible references and pick the most stable one (active site)
        # Generate slab reference to place the adsorbates
        self.ref_slabs, self.reactive_idx = self._get_reference_slab()

        # Mxide method
        # self.mxidegen = self._mxidegen() #FIXME: Not needed for metals
        if not self.surface_coverage[0] == "clean":

            # breakpoint()
            # TODO: Logic for metals -> can just pick the site from the oxygen
            self.selected_sites = [self.slab[idx].coords for idx in self.reactive_idx]

        #    # Shifted bulk_like_sites
        #    self.bulk_like_dict = self._get_shifted_bulk_like_sites()

        #    # Selected site
        #    self.selected_site = self.bulk_like_dict[self.reactive_idx]
        else:  # clean coverage
            self.selected_sites = [self.bulk_like_sites[self.reactive_idx[0]]]

        # np seed
        np.random.seed(self.random_state)

    def _get_surface_termination(self):
        """Helper function to get whether the surface is OH or Ox terminated"""

        # FIXME: Refer to the slab_orig and align it to the relaxed one by calling slab.sort()
        self.slab_orig.sort()
        self.slab.sort()
        # Get the indices of the adsorbates from the slab_orig
        termination_indices = np.where(
            np.array(self.slab_orig.site_properties["surface_properties"])
            == "adsorbate"
        )[0].tolist()
        termination_info = [
            [idx, self.slab[idx].specie, self.slab[idx].frac_coords]
            for idx in termination_indices
            # if "surface_properties" in site.properties
        ]

        # Filter sites information
        ads_species = [site[1] for site in termination_info]
        ads_indices = [site[0] for site in termination_info]

        # OH or Ox coverage
        surface_coverage = ["oxo" if Element("H") not in ads_species else "oh"]

        if len(termination_info) == 0:  # clean termination
            surface_coverage = ["clean"]
            return surface_coverage, ads_species, ads_indices, termination_info

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

    def _find_nearest_hydrogen(self, site_idx):
        """Depending on how the surface atoms are sorted we need to find the nearest H"""
        radius = 1
        neighbors = self.slab.get_neighbors(self.slab[site_idx], radius)
        while not any([site.species_string in ["H"] for site in neighbors]):
            radius += 0.1
            neighbors = [
                nn for nn in self.slab.get_neighbors(self.slab[site_idx], radius)
            ]

        # fixed_site = [site for idx, site in enumerate(self.slab) if idx == site_idx][0]

        # min_dist = np.inf
        # for site in search_list:
        #    dist = np.linalg.norm(fixed_site.frac_coords - site[2])
        #    if dist <= min_dist:
        #        min_dist = dist
        #        nn_site = site
        return [site for site in neighbors if site.species_string in ["H"]][0]

    def _find_nearest_metal(self, reactive_idx):
        """Find reactive site by min distance between any metal and oxygen"""
        radius = 1
        neighbors = self.slab.get_neighbors(self.slab[reactive_idx], radius)
        while not any([site.species_string not in ["O", "H"] for site in neighbors]):
            radius += 0.1
            neighbors = [
                nn for nn in self.slab.get_neighbors(self.slab[reactive_idx], radius)
            ]

        return [site for site in neighbors if site.species_string not in ["O", "H"]][0]

    def _group_ads_sites_by_metal(self):
        """Groups self.ads_indices by metal"""
        sites_by_metal = {}
        for ads_idx in self.ads_indices:
            close_site = self._find_nearest_metal(ads_idx)
            if str(close_site.specie) not in sites_by_metal.keys():
                sites_by_metal.update({str(close_site.specie): [ads_idx]})
            elif str(close_site.specie) in sites_by_metal.keys():
                sites_by_metal[str(close_site.specie)].append(ads_idx)
        if self.surface_coverage[0] == "clean":
            end_idx = np.where(
                self.slab_clean.frac_coords[:, 2] >= self.slab_clean.center_of_mass[2]
            )[0][-1]
            metal_idx = []
            for bulk_idx, bulk_like_site in enumerate(self.bulk_like_sites):
                min_dist = np.inf  # initialize the min_dist register
                min_metal_idx = 0  # initialize the min_ox_idx
                for idx, site in enumerate(self.slab_clean):
                    if (
                        site.species_string != "O"  # FIXME
                        and site.frac_coords[2] > self.slab_clean.center_of_mass[2]
                    ):  # metal
                        dist = np.linalg.norm(bulk_like_site - site.coords)
                        if dist < min_dist:
                            min_dist = dist  # update the dist register
                            min_metal_idx = idx  # update the idx register
                            min_specie = site.species_string
                    if (
                        idx == end_idx
                    ):  # make sure that len(bulk_like_sites) == len(ox_idx)
                        metal_idx.append(min_metal_idx)
                        if min_specie not in sites_by_metal:
                            sites_by_metal[min_specie] = []
                            sites_by_metal[min_specie].append((min_metal_idx, bulk_idx))
                        else:
                            sites_by_metal[min_specie].append((min_metal_idx, bulk_idx))

        return sites_by_metal

    def _get_reference_slab(self):
        """Random selects a termination adsorbate and clean its"""

        if self.surface_coverage[0] == "oxo":
            ref_slab = self.slab.copy()
            # Orig magmoms for the adslabs
            ref_slab.add_site_property(
                "magmom", self.slab_orig.site_properties["magmom"]
            )
            if self.checkpoint_path:  # streamline
                ref_slabs = []
                for active_site in self.ads_indices:
                    ref_slab_copy = ref_slab.copy()
                    ref_slab_copy.remove_sites(indices=[active_site])
                    ref_slabs.append(ref_slab_copy)
                # ref_slab, stable_index = find_most_stable_config(
                #    ref_slabs, self.checkpoint_path
                # ) # We generate a separate OER branch for each possible site
                ref_slabs = [
                    Slab(
                        ref_slab.lattice,
                        ref_slab.species,
                        ref_slab.frac_coords,
                        miller_index=ref_slabs[0].miller_index,
                        oriented_unit_cell=ref_slabs[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=ref_slab.site_properties,
                    )
                    for ref_slab in ref_slabs
                ]
            else:  # random
                reactive_site = np.random.choice(self.ads_indices)
                ref_slab.remove_sites(indices=[reactive_site])

            return ref_slabs, self.ads_indices

        elif self.surface_coverage[0] == "oh":
            ref_slab = self.slab.copy()
            # Orig magmoms for the adslabs
            ref_slab.add_site_property(
                "magmom", self.slab_orig.site_properties["magmom"]
            )
            ads_indices_oxygen = [
                site[0] for site in self.termination_info if site[1] == Element("O")
            ]
            ads_indices_hyd = [
                site for site in self.termination_info if site[1] == Element("H")
            ]
            if self.checkpoint_path:  # streamline
                ref_slabs = []
                reactive_sites = []
                oxygens_on_target = [
                    x for x in self.ads_indices if x in ads_indices_oxygen
                ]
                for active_site in oxygens_on_target:
                    ref_slab_copy = ref_slab.copy()
                    hyd_site = self._find_nearest_hydrogen(active_site)
                    reactive_site = [active_site, hyd_site.index]
                    ref_slab_copy.remove_sites(indices=reactive_site)
                    reactive_sites.append(reactive_site)
                    ref_slabs.append(ref_slab_copy)
                # ref_slab, stable_index = find_most_stable_config(
                #    ref_slabs, self.checkpoint_path
                # )
                ref_slabs = [
                    Slab(
                        ref_slab.lattice,
                        ref_slab.species,
                        ref_slab.frac_coords,
                        miller_index=ref_slabs[0].miller_index,
                        oriented_unit_cell=ref_slabs[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=ref_slab.site_properties,
                    )
                    for ref_slab in ref_slabs
                ]
                # ref_slab = Slab(
                #    ref_slab.lattice,
                #    ref_slab.species,
                #    ref_slab.frac_coords,
                #    miller_index=ref_slabs[0].miller_index,
                #    oriented_unit_cell=ref_slabs[0].oriented_unit_cell,
                #    shift=0,
                #    scale_factor=0,
                #    site_properties=ref_slab.site_properties,
                # )
                # reactive_site = reactive_sites[stable_index]
                # reactive_site_oxygen = oxygens_on_target[stable_index]
            else:
                reactive_site_oxygen = np.random.choice(ads_indices_oxygen)
                hyd_site = self._find_nearest_hydrogen(reactive_site_oxygen)
                reactive_site = [reactive_site_oxygen, hyd_site]
                ref_slab.remove_sites(indices=reactive_site)
            return ref_slabs, oxygens_on_target
        else:  # clean termination?
            ref_slab = self.slab_clean.copy()
            # Orig magmoms for the adslabs
            ref_slab.add_site_property(
                "magmom", self.slab_orig.site_properties["magmom"]
            )
            reactive_idx = np.random.choice(np.arange(len(self.bulk_like_sites)))
            reactive_site = self.bulk_like_sites[
                np.random.choice(np.arange(len(self.bulk_like_sites)))
            ]
            return [ref_slab], [reactive_idx]

    def _mxidegen(self, repeat=[1, 1, 1], verbose=False):
        """Returns the MXide Method for the ref_slab"""
        mxidegen = MXideAdsorbateGenerator(
            self.slab_clean,
            repeat=repeat,
            verbose=verbose,
            positions=["MX_adsites"],
            relax_tol=0.025,
        )
        return mxidegen

    def _get_shifted_bulk_like_sites(self):
        """Get Perturbed bulk-like sites"""
        # Bondlength and X-specie from mxide method
        _, X = self.mxidegen.bondlengths_dict, self.mxidegen.X

        # Perturb pristine bulk_like sites {idx: np.array([x,y,z])}
        bulk_like_shifted_dict = self._bulk_like_adsites_perturbation_oxygens(
            self.slab_orig, self.slab, X=X
        )

        return bulk_like_shifted_dict

    def _bulk_like_adsites_perturbation(
        self, slab_ref, slab, bulk_like_sites, bondlength, X
    ):
        """Let's perturb bulk_like_sites with delta (xyz)"""
        slab_ref_coords = slab_ref.cart_coords
        slab_coords = slab.cart_coords

        delta_coords = slab_coords - slab_ref_coords

        metal_idx = []
        for bulk_like_site in bulk_like_sites:
            for idx, site in enumerate(slab_ref):
                if (
                    site.specie != Element(X)
                    and site.coords[2] > slab_ref.center_of_mass[2]
                ):
                    dist = np.linalg.norm(bulk_like_site - site.coords)
                    if dist < bondlength:
                        metal_idx.append(idx)

        bulk_like_deltas = [delta_coords[i] for i in metal_idx]
        return [n + m for n, m in zip(bulk_like_sites, bulk_like_deltas)]

    def _bulk_like_adsites_perturbation_oxygens(self, slab_ref, slab, X):
        """peturbation on oxygens"""
        slab_ref_coords = slab_ref.cart_coords  # input
        slab_coords = slab.cart_coords  # output
        end_idx = np.where(slab_ref.frac_coords[:, 2] >= slab_ref.center_of_mass[2])[0][
            -1
        ]

        delta_coords = slab_coords - slab_ref_coords

        ox_idx = []
        for bulk_like_site in self.bulk_like_sites:
            min_dist = np.inf  # initialize the min_dist register
            min_ox_idx = 0  # initialize the min_ox_idx
            for idx, site in enumerate(slab_ref):
                if (
                    site.specie == Element(X)
                    and site.frac_coords[2] > slab_ref.center_of_mass[2]
                ):
                    dist = np.linalg.norm(bulk_like_site - site.coords)
                    if dist < min_dist:
                        min_dist = dist  # update the dist register
                        min_ox_idx = idx  # update the idx register
                if idx == end_idx:  # make sure that len(bulk_like_sites) == len(ox_idx)
                    ox_idx.append(min_ox_idx)

        bulk_like_deltas = [delta_coords[i] for i in ox_idx]
        bulk_like_shifted = [
            n + m for n, m in zip(self.bulk_like_sites, bulk_like_deltas)
        ]
        return {k: [v] for (k, v) in zip(ox_idx, bulk_like_shifted)}

    def _get_clean_slab(self):
        """Remove all the adsorbates"""
        clean_slab = self.slab_orig.copy()
        clean_slab.remove_sites(indices=self.all_ads_indices)
        return clean_slab

    def _get_oer_intermediates(
        self, adsorbate, suffix=None, axis_rotation=[0, 0, 1], n_rotations=32
    ):
        """Returns OH/Ox/OOH intermediates"""
        # Adsorbate manipulation
        adsorbate = adsorbate.copy()
        adsorbate_label = "".join([site.species_string for site in adsorbate])
        adsorbate_angles = self._get_angles(n_rotations=n_rotations)

        intermediates_dict = {}
        # FIXME: Logic for metals can use the internal rotation on the slab
        reference_dict = {}
        for ref_slab, selected_site, idx in zip(
            self.ref_slabs, self.selected_sites, self.reactive_idx
        ):
            reference_dict[f"reference_{idx}"] = ref_slab.as_dict()
            ads_slab = ref_slab.copy()
            asf = AdsorbateSiteFinder(ads_slab)
            slab_ads = asf.add_adsorbate(adsorbate, selected_site)
            intermediates_dict[f"{idx}"] = {}
            # Now rotate about the binding oxo position in case the adsorbate has rot deg freedom
            if len(adsorbate) > 1:
                binding_site_index = np.where(
                    np.array(slab_ads.site_properties["binding_site"]) == True
                )[0].tolist()[-1]
                rotate_site_indices = np.where(
                    np.array(slab_ads.site_properties["surface_properties"])
                    == "adsorbate"
                )[0].tolist()[-len(adsorbate) :]
                for i, ang in enumerate(adsorbate_angles):
                    slab_ads.rotate_sites(
                        rotate_site_indices,
                        ang,
                        axis_rotation,
                        slab_ads[binding_site_index].coords,
                        to_unit_cell=False,
                    )
                    slab_ads_snap = slab_ads.copy()
                    intermediates_dict[f"{idx}"].update(
                        {f"{adsorbate_label}_{suffix}_{i}": slab_ads_snap.as_dict()}
                    )
            else:
                intermediates_dict[f"{idx}"].update(
                    {f"{adsorbate_label}_0": slab_ads.as_dict()}
                )

            # adsorbate_rotations = self.mxidegen.get_transformed_molecule(
            #    adsorbate, axis=axis_rotation, angles_list=adsorbate_angles
            # )
            # Intermediates generation
            # intermediates_dict = {}
            # for ads_rot_idx in range(len(adsorbate_rotations)):
            #    ads_slab = self.ref_slab.copy()
            #    ads_slab = self._add_adsorbates(
            #        ads_slab, self.selected_site, adsorbate_rotations[ads_rot_idx]
            #    )
            #    if suffix:
            #        intermediates_dict.update(
            #            {f"{adsorbate_label}_{suffix}_{ads_rot_idx}": ads_slab.as_dict()}
            #        )
            #    else:
            #        intermediates_dict.update(
            #            {f"{adsorbate_label}_{ads_rot_idx}": ads_slab.as_dict()}
            #        )

        return reference_dict, intermediates_dict

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
        # for ref_slab in self.ref_slabs:
        #    reference_slab = ref_slab.as_dict()
        #    reference_dict = {"reference": reference_slab}

        # Generate intermediantes depeding on Termination
        if self.surface_coverage[0] == "oxo":
            ref_dict, oh_intermediates = self._get_oer_intermediates(
                self.adsorbates["OH"]
            )
            oh_int_stab = {}
            ooh_int_stab = {}
            for site in oh_intermediates:
                if self.checkpoint_path:
                    configs = [
                        Slab.from_dict(oh_intermediates[site][k])
                        for k in oh_intermediates[site].keys()
                    ]
                    slab_ads, slab_index = find_most_stable_config(
                        configs, self.checkpoint_path
                    )
                    # Cast the Struct to Slab so can add metadata
                    slab_ads = Slab(
                        slab_ads.lattice,
                        slab_ads.species,
                        slab_ads.frac_coords,
                        miller_index=configs[0].miller_index,
                        oriented_unit_cell=configs[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=slab_ads.site_properties,
                    )
                    oh_int_stab[f"OH_{site}_{slab_index.item()}"] = slab_ads.as_dict()
                _, ooh_up = self._get_oer_intermediates(
                    self.adsorbates["OOH_up"], suffix="up"
                )
                _, ooh_down = self._get_oer_intermediates(
                    self.adsorbates["OOH_down"], suffix="down"
                )
                if self.checkpoint_path:
                    # Can commingle the OOH and pick only one
                    ooh_intermediates = {**ooh_down[site], **ooh_up[site]}
                    configs = [
                        Slab.from_dict(ooh_intermediates[k])
                        for k in ooh_intermediates.keys()
                    ]
                    slab_ads, slab_index = find_most_stable_config(
                        configs, self.checkpoint_path
                    )
                    slab_ads = Slab(
                        slab_ads.lattice,
                        slab_ads.species,
                        slab_ads.frac_coords,
                        miller_index=configs[0].miller_index,
                        oriented_unit_cell=configs[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=slab_ads.site_properties,
                    )
                    ooh_int_stab[f"OOH_{site}_{slab_index.item()}"] = slab_ads.as_dict()
            oer_intermediates = {
                **ref_dict,
                **oh_int_stab,
                **ooh_int_stab,
            }
            # else:
            #    oer_intermediates = {
            #        **reference_dict,
            #        **oh_intermediates,
            #        **ooh_up,
            #        **ooh_down,
            #    }

            return oer_intermediates

        elif self.surface_coverage[0] == "oh":
            ref_dict, ox_intermediates = self._get_oer_intermediates(
                self.adsorbates["Ox"], n_rotations=1
            )
            _, ooh_up = self._get_oer_intermediates(
                self.adsorbates["OOH_up"], suffix="up"
            )
            _, ooh_down = self._get_oer_intermediates(
                self.adsorbates["OOH_down"], suffix="down"
            )
            ox_int_stab = {}
            ooh_int_stab = {}
            for site in ox_intermediates:
                if self.checkpoint_path:
                    configs_ox = [Slab.from_dict(ox_intermediates[site]["O_0"])]
                    slab_ads_ox, slab_index_o = find_most_stable_config(
                        configs_ox, self.checkpoint_path
                    )  # just so we can decorate with tags
                    slab_ads_o = Slab(
                        slab_ads_ox.lattice,
                        slab_ads_ox.species,
                        slab_ads_ox.frac_coords,
                        miller_index=configs_ox[0].miller_index,
                        oriented_unit_cell=configs_ox[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=slab_ads_ox.site_properties,
                    )
                    ox_int_stab[f"O_{site}_0"] = slab_ads_o.as_dict()
                    ooh_intermediates = {**ooh_down[site], **ooh_up[site]}
                    configs = [
                        Slab.from_dict(ooh_intermediates[k])
                        for k in ooh_intermediates.keys()
                    ]
                    slab_ads, slab_index = find_most_stable_config(
                        configs, self.checkpoint_path
                    )
                    slab_ads = Slab(
                        slab_ads.lattice,
                        slab_ads.species,
                        slab_ads.frac_coords,
                        miller_index=configs[0].miller_index,
                        oriented_unit_cell=configs[0].oriented_unit_cell,
                        shift=0,
                        scale_factor=0,
                        site_properties=slab_ads.site_properties,
                    )
                    # ooh_intermediates = {f"OOH_{slab_index}": slab_ads.as_dict()}
                    ooh_int_stab[f"OOH_{site}_{slab_index.item()}"] = slab_ads.as_dict()
            oer_intermediates = {
                **ref_dict,
                **ox_int_stab,
                **ooh_int_stab,
            }
            # breakpoint()
            # else:
            #    oer_intermediates = {
            #        **reference_dict,
            #        **ox_intermediates,
            #        **ooh_up,
            #        **ooh_down,
            #    }
        else:  # clean termination
            ref_slab, ox_intermediates = self._get_oer_intermediates(
                self.adsorbates["Ox"], n_rotations=1
            )
            _, oh_intermediates = self._get_oer_intermediates(self.adsorbates["OH"])
            _, ooh_up = self._get_oer_intermediates(
                self.adsorbates["OOH_up"], suffix="up"
            )
            _, ooh_down = self._get_oer_intermediates(
                self.adsorbates["OOH_down"], suffix="down"
            )
            configs_ox = [
                Slab.from_dict(ox_intermediates[str(self.reactive_idx[0])]["O_0"])
            ]
            slab_ads_ox, slab_index_o = find_most_stable_config(
                configs_ox, self.checkpoint_path
            )  # just so we can decorate with tags
            slab_ads_o = Slab(
                slab_ads_ox.lattice,
                slab_ads_ox.species,
                slab_ads_ox.frac_coords,
                miller_index=configs_ox[0].miller_index,
                oriented_unit_cell=configs_ox[0].oriented_unit_cell,
                shift=0,
                scale_factor=0,
                site_properties=slab_ads_ox.site_properties,
            )
            ox_int_stab = {}
            ox_int_stab[f"O_{self.reactive_idx[0]}_0"] = slab_ads_o.as_dict()
            if self.checkpoint_path:
                configs = [
                    Slab.from_dict(oh_intermediates[str(self.reactive_idx[0])][k])
                    for k in oh_intermediates[str(self.reactive_idx[0])].keys()
                ]
                slab_ads, slab_index = find_most_stable_config(
                    configs, self.checkpoint_path
                )
                # Cast the Struct to Slab so can add metadata
                slab_ads = Slab(
                    slab_ads.lattice,
                    slab_ads.species,
                    slab_ads.frac_coords,
                    miller_index=configs[0].miller_index,
                    oriented_unit_cell=configs[0].oriented_unit_cell,
                    shift=0,
                    scale_factor=0,
                    site_properties=slab_ads.site_properties,
                )

                oh_intermediates = {
                    f"OH_{self.reactive_idx[0]}_{slab_index.item()}": slab_ads.as_dict()
                }
                ooh_intermediates = {
                    **ooh_down[str(self.reactive_idx[0])],
                    **ooh_up[str(self.reactive_idx[0])],
                }
                configs = [
                    Slab.from_dict(ooh_intermediates[k])
                    for k in ooh_intermediates.keys()
                ]
                slab_ads, slab_index = find_most_stable_config(
                    configs, self.checkpoint_path
                )
                slab_ads = Slab(
                    slab_ads.lattice,
                    slab_ads.species,
                    slab_ads.frac_coords,
                    miller_index=configs[0].miller_index,
                    oriented_unit_cell=configs[0].oriented_unit_cell,
                    shift=0,
                    scale_factor=0,
                    site_properties=slab_ads.site_properties,
                )
                ooh_intermediates = {
                    f"OOH_{self.reactive_idx[0]}_{slab_index.item()}": slab_ads.as_dict()
                }
                oer_intermediates = {
                    **ref_slab,
                    **ox_int_stab,
                    **oh_intermediates,
                    **ooh_intermediates,
                }
            else:
                oer_intermediates = {
                    **reference_dict,
                    **ox_intermediates,
                    **oh_intermediates,
                    **ooh_up,
                    **ooh_down,
                }

        return oer_intermediates
