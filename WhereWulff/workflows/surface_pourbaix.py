from __future__ import absolute_import, division, print_function, unicode_literals

import uuid

import numpy as np
import torch
from ase.constraints import FixAtoms
from atomate.vasp.config import DB_FILE, VASP_CMD
from fireworks import Workflow
from ocpmodels.common.utils import setup_imports
from ocpmodels.trainers import ForcesTrainer
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from scipy.spatial.distance import pdist, squareform
from WhereWulff.adsorption.MXide_adsorption import MXideAdsorbateGenerator
from WhereWulff.fireworks.optimize import AdsSlab_FW
from WhereWulff.fireworks.surface_pourbaix import SurfacePBX_FW
from WhereWulff.workflows.oer import OER_WF
from WhereWulff.utils import find_most_stable_config
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


class SurfaceCoverageML(object):
    """
    Given a surface structure and an adsorbate,
    predict the most stable termination using OCP models.

    Args:
        slab             (pmg struct)  : Pymatgen Structure object of a clean slab model.
        slab_ref         (pmg struct)  : Pymatgen Structure object origin of the slab after optimization.
        adsorbate        (pmg molecule): Adsorbate to be placed as surface coverage, assuming full monolayer.
        model_checkpoint (env)         : Path to the best OCP model checkpoint (my_fworker.yml variable).
        model_config     (env)         : Path to the OCP model configuration file (my_fworker.yml variable).
        to_db            (bool)        : If results should be stored into db or local json file.
        db_file          (env)         : Environment variable to connect to the DB.

    Returns:
        The most stable surface coverage, using ML.
    """

    def __init__(self, slab_ref, slab, adsorbate, is_metal, checkpoint_path):

        self.checkpoint_path = checkpoint_path
        # Init Mxide
        if not is_metal:  # MXide
            mxidegen = MXideAdsorbateGenerator(
                slab_ref, repeat=[1, 1, 1], verbose=False, positions=["MX_adsites"]
            )
            mxidegen.slab.sort()
            surface_prop = mxidegen.slab.site_properties["surface_properties"]
            slab.sort()
            slab.add_site_property("surface_properties", surface_prop)

            self.bulk_like, _ = mxidegen.get_bulk_like_adsites()
            _, self.X = mxidegen.bondlengths_dict, mxidegen.X
        else:  # Metals
            asf = AdsorbateSiteFinder(slab_ref)
            self.bulk_like = asf.find_adsorption_sites(
                positions=["bridge"], put_inside=True, symm_reduce=0
            )["all"]
            self.X = "not_oxide"
        # Make sure that the sites are in the corresponding order between the slab_ref and the relaxed slab
        # Perturb bulk_like_sites in case the slab model is optimized
        slab_ref.sort()

        # Perturb bulk_like_sites in case the slab model is optimized
        self.bulk_like_shifted = self._bulk_like_adsites_perturbation(
            slab_ref=slab_ref, slab=slab
        )

        # Set n_rotations
        n = len(adsorbate[0]) if type(adsorbate).__name__ == "list" else len(adsorbate)
        n_rotations = 1 if n == 1 else 8

        # Adsorbate formula
        adsorbate_comp = adsorbate.composition.as_dict()
        self.adsorbate_formula = "".join(adsorbate_comp.keys())

        # if is_metal and n_rotations == 1:
        #    slab_ads = slab.copy()
        #    for site_pos in self.bulk_like_shifted:
        #        asf = AdsorbateSiteFinder(slab_ads)
        #        slab_ads = asf.add_adsorbate(adsorbate, site_pos)
        #    slab_ads = Slab(
        #        slab_ads.lattice,
        #        slab_ads.species,
        #        slab_ads.frac_coords,
        #        miller_index=slab.miller_index,
        #        oriented_unit_cell=slab.oriented_unit_cell,
        #        shift=0,
        #        scale_factor=0,
        #        site_properties=slab_ads.site_properties,
        #    )
        #    self.pmg_stable_config = slab_ads.copy()
        #    return

        # Get angles
        self.angles = self._get_angles(n_rotations=n_rotations)

        # Rotate adsorbate
        # self.adsorbate_rotations = mxidegen.get_transformed_molecule_MXides(
        #    adsorbate, axis=[0, 0, 1], angles_list=self.angles
        # )

        # (pseudo)-Randomly pick the first site
        site_index = np.random.choice(len(self.bulk_like_shifted))
        remaining_site_indices = list(range(len(self.bulk_like_shifted)))

        # Place the 1st adsorbate on that site
        square_distance_matrix = self._get_closest_neighbors(self.bulk_like_shifted)[1]
        # Loop
        counter = 0
        occ_site_indices = []
        while remaining_site_indices:
            if len(remaining_site_indices) == len(self.bulk_like_shifted):
                slab_ads = slab.copy()
                previous_site_1, previous_site_2 = None, None
                original_sdm = None
            else:
                previous_site_1, previous_site_2 = site_1, site_2

            (
                site_1,
                site_2,
                remaining_site_indices,
                square_distance_matrix,
                original_sdm,
                occ_site_indices,
            ) = self.find_and_update_sites(
                square_distance_matrix,
                remaining_site_indices,
                previous_site_1,
                previous_site_2,
                original_sdm,
                occ_site_indices,
                slab_ads,
            )
            if site_1 is None and site_2 is None:
                break
            if not is_metal:
                for site in [site_1, site_2]:
                    if site is not None:
                        slab_ads = add_adsorbates(
                            slab_ads.copy(),
                            [self.bulk_like_shifted[site]],
                            adsorbate,
                        )
            else:
                for site in [site_1, site_2]:
                    if site is not None:
                        asf = AdsorbateSiteFinder(slab_ads)
                        slab_ads = asf.add_adsorbate(
                            adsorbate, self.bulk_like_shifted[site]
                        )
            counter += 1
            if is_metal and n_rotations == 1:
                continue  # There is no point rotating e.g Ox

            configs = self.rotate_site_indices(
                slab_ads,
                counter,
                single_site=True if site_2 is None and site_1 is not None else False,
            )
            slab_ads, index = find_most_stable_config(
                configs, checkpoint_path=self.checkpoint_path
            )
        # Cast the structure into a Slab object
        slab_ads = Slab(
            slab_ads.lattice,
            slab_ads.species,
            slab_ads.frac_coords,
            miller_index=slab.miller_index,
            oriented_unit_cell=slab.oriented_unit_cell,
            shift=0,
            scale_factor=0,
            site_properties=slab_ads.site_properties,
        )
        slab_ads.to(filename="POSCAR_most_stable")
        self.pmg_stable_config = slab_ads.copy()

    def find_and_update_sites(
        self,
        square_distance_matrix,
        remaining_site_indices,
        previous_site_1,
        previous_site_2,
        original_sdm,
        occ_site_indices,
        slab_ads,
    ):  # FIXME: Is there a way to project to reciprocal space so the pairwise distances factor in mirror periodic images or is it enough to
        # model large supercells
        # We need to make sure we locate sites that are close to periodic images of previously placed
        # adsorbates
        if len(remaining_site_indices) < len(self.bulk_like_shifted):
            periodic_neighbors = [
                slab_ads.get_sites_in_sphere(self.bulk_like_shifted[i], 2)
                for i in remaining_site_indices
            ]
            periodic_mask = [
                "O" in [y.species_string for y in x]
                and "H" in [y.species_string for y in x]
                for x in periodic_neighbors
            ]
            indices_to_exclude = np.array(remaining_site_indices)[
                np.array(periodic_mask)
            ]
            for ind in indices_to_exclude:
                square_distance_matrix[ind, :] = np.inf
                square_distance_matrix[:, ind] = np.inf
                remaining_site_indices.remove(ind)

        # Let's constrain the sites to be at least 1 Angs apart since this is the average bond length
        # of an O-H bond

        if original_sdm is None:
            original_sdm = square_distance_matrix.copy()
        if len(remaining_site_indices) > 2:
            square_distance_matrix = np.ma.array(
                square_distance_matrix, mask=square_distance_matrix < 2.3
            )
            square_distance_matrix = square_distance_matrix.filled(np.inf)
            if previous_site_1 is not None and previous_site_2 is not None:
                mask = (
                    original_sdm[
                        np.ix_(
                            np.arange(original_sdm.shape[0]),
                            [previous_site_1, previous_site_2],
                        )
                    ]
                    > 2.3
                )
                indices = np.where(mask == False)[
                    0
                ]  # Replace with inf the rows, columns yielded since these are too close to the previous index
                if indices.size > 0:
                    square_distance_matrix[indices] = np.inf
                    square_distance_matrix[:, indices] = np.inf

            if (square_distance_matrix == np.inf).all():
                return (
                    None,
                    None,
                    remaining_site_indices,
                    square_distance_matrix,
                    original_sdm,
                    occ_site_indices,
                )

            row, column = np.unravel_index(  # Pick next site
                square_distance_matrix.argmin(), square_distance_matrix.shape
            )
        elif (
            len(remaining_site_indices) <= 2 and len(remaining_site_indices) >= 0
        ):  # FIXME: Need to check whether we can just pick both or one of them # Need to place one by one
            mask = (
                original_sdm[
                    np.ix_(
                        occ_site_indices,
                        # [previous_site_1, previous_site_2],
                        remaining_site_indices,
                    )
                ]
                > 2.0
            )  # Identify if both sites are eligible
            vmask = np.all(mask, axis=0)
            indices = np.where(vmask == True)[0]
            if (
                indices.size > 1
            ):  # Pick two #TODO: We pick one even if there are two to protect against periodic clashes
                row = np.array(remaining_site_indices)[indices].tolist()[0]
                column = None
            elif indices.size == 1:  # Pick one
                row = np.array(remaining_site_indices)[indices].tolist()[0]
                column = None
            else:
                return (
                    None,
                    None,
                    [],
                    square_distance_matrix,
                    original_sdm,
                    occ_site_indices,
                )
        # elif len(remaining_site_indices) == 2:
        #    row, column = remaining_site_indices
        # elif (
        #    len(remaining_site_indices) == 1
        # ):  # Means we started with an odd number of sites
        #    row, column = remaining_site_indices[0], None
        # else:
        #    return None, None, remaining_site_indices, square_distance_matrix
        for value in [row, column]:
            if value is not None:
                occ_site_indices.append(
                    remaining_site_indices.pop(remaining_site_indices.index(value))
                )
        if row is not None:
            square_distance_matrix[row, :] = np.inf
            square_distance_matrix[:, row] = np.inf
        if column is not None:
            square_distance_matrix[:, column] = np.inf
            square_distance_matrix[column, :] = np.inf
        return (
            row,
            column,
            remaining_site_indices,
            square_distance_matrix,
            original_sdm,
            occ_site_indices,
        )

    # def find_and_update_sites(self, square_distance_matrix, remaining_site_indices):
    #    row, column = np.unravel_index(
    #        square_distance_matrix.argmin(), square_distance_matrix.shape
    #    )
    #    for index in [row, column]:
    #        remaining_site_indices.remove(index)
    #    square_distance_matrix[row, :] = np.inf
    #    square_distance_matrix[:, column] = np.inf
    #    square_distance_matrix[column, :] = np.inf
    #    square_distance_matrix[:, row] = np.inf
    #    return (row, column, remaining_site_indices, square_distance_matrix)
    def rotate_site_indices(self, slab_ads, counter, single_site=False):
        rotate_site_indices = np.where(
            (np.array(slab_ads.site_properties["binding_site"]) == True)
        )[0].tolist()[-2:]
        configs = []
        # for site, other_site in rotate_site_indices:
        for i, ang in enumerate(self.angles):
            if len(configs) > 1 and not single_site:
                slab_ads = configs[len(configs) - len(self.angles)].copy()
            first_site = rotate_site_indices[0]
            slab_ads.rotate_sites(
                [first_site, first_site + 1],
                ang,
                [0, 0, 1],
                slab_ads[first_site].coords,
                to_unit_cell=False,
            )
            if single_site:
                configs.append(slab_ads.copy())
            # configs.append(slab_ads.copy())
            else:
                second_site = rotate_site_indices[1]
                for ang2 in self.angles:
                    slab_ads.rotate_sites(
                        [second_site, second_site + 1],
                        ang2,
                        [0, 0, 1],
                        slab_ads[second_site].coords,
                        to_unit_cell=False,
                    )
                    slab_ads.to(filename=f"POSCAR_dimer_{counter}_{ang}_{ang2}")
                    configs.append(slab_ads.copy())
        return configs

    # def rotate_site_indices(self, slab_ads, counter, adsorbate, single_site):
    #    anchor_site_indices = np.where(
    #        (np.array(slab_ads.site_properties["binding_site"]) == True)
    #    )[0].tolist()[-2:]
    #    adsorbate_indices_two = np.where(
    #        (np.array(slab_ads.site_properties["surface_properties"]) == "adsorbate")
    #    )[0].tolist()[-len(adsorbate) :]
    #    adsorbate_indices_one = np.where(
    #        (np.array(slab_ads.site_properties["surface_properties"]) == "adsorbate")
    #    )[0].tolist()[-2 * len(adsorbate) : -len(adsorbate)]

    #    configs = []
    #    # for site, other_site in rotate_site_indices:
    #    for i, ang in enumerate(self.angles):
    #        if len(configs) > 1:
    #            slab_ads = configs[len(configs) - len(self.angles)].copy()
    #        first_site = anchor_site_indices[0]
    #        slab_ads.rotate_sites(
    #            adsorbate_indices_one,
    #            ang,
    #            [0, 0, 1],
    #            slab_ads[first_site].coords,
    #            to_unit_cell=False,
    #        )
    #        # configs.append(slab_ads.copy())
    #        second_site = anchor_site_indices[1]
    #        for ang2 in self.angles:
    #            slab_ads.rotate_sites(
    #                adsorbate_indices_two,
    #                ang2,
    #                [0, 0, 1],
    #                slab_ads[second_site].coords,
    #                to_unit_cell=False,
    #            )
    #            # slab_ads.to(filename=f"visuals/POSCAR_dimer_{counter}_{i}_{ang}_{ang2}")
    #            configs.append(slab_ads.copy())

    #    return configs

    def _get_angles(self, n_rotations=8):
        """Get angles"""
        angles = []
        for i in range(n_rotations):
            deg = (2 * np.pi / n_rotations) * i
            angles.append(deg)
        return angles

    def _get_closest_neighbors(self, sites):
        """Get distances between bulk_like sites"""
        sites_stack = np.vstack(sites)
        distances = squareform(pdist(sites_stack, "euclidean"))
        np.fill_diagonal(distances, np.inf)
        site_pairs = list(
            zip(range(len(self.bulk_like_shifted)), distances.argmin(axis=0))
        )
        return site_pairs, distances

    def _bulk_like_adsites_perturbation(self, slab_ref, slab):
        """Let's perturb bulk_like_sites with delta (x,y,z) comparing input and output"""
        slab_ref_coords = slab_ref.cart_coords
        slab_coords = slab.cart_coords

        delta_coords = slab_coords - slab_ref_coords

        metal_idx = []
        for bulk_like_site in self.bulk_like:
            min_dist = np.inf  # initialize min_dist register
            min_metal_idx = 0
            end_idx = np.where(
                slab_ref.frac_coords[:, 2] >= slab_ref.center_of_mass[2]
            )[0][-1]
            for idx, site in enumerate(slab_ref):
                if (
                    site.species_string != self.X
                    and site.frac_coords[2] > slab_ref.center_of_mass[2]
                ):
                    dist = np.linalg.norm(bulk_like_site - site.coords)

                    if dist < min_dist:
                        min_dist = dist
                        min_metal_idx = idx

                if idx == end_idx:
                    metal_idx.append(min_metal_idx)

        bulk_like_deltas = [delta_coords[i] for i in metal_idx]
        return [n + m for n, m in zip(self.bulk_like, bulk_like_deltas)]


# Angles list
def get_angles(n_rotations=4):
    """Get angles like in the past"""
    angles = []
    for i in range(n_rotations):
        deg = (2 * np.pi / n_rotations) * i
        angles.append(deg)
    return angles


def add_adsorbates(adslab, ads_coords, molecule, z_offset=[0, 0, 0.15]):
    """Add molecule in all ads_coords once"""
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

    # Fill in the None values in the adslab because of addition of adsorbate
    adslab.add_site_property(
        "tags", [2 if x is None else x for x in adslab.site_properties["tags"]]
    )
    adslab.add_site_property(
        "bulk_wyckoff",
        ["None" if x is None else x for x in adslab.site_properties["bulk_wyckoff"]],
    )
    adslab.add_site_property(
        "forces",
        ["None" if x is None else x for x in adslab.site_properties["forces"]],
    )
    adslab.add_site_property(
        "binding_site",
        [False if x is None else x for x in adslab.site_properties["binding_site"]],
    )
    adslab.add_site_property(
        "selective_dynamics",
        [
            [True, True, True] if x is None else x
            for x in adslab.site_properties["selective_dynamics"]
        ],
    )
    adslab.add_site_property(
        "bulk_equivalent",
        ["None" if x is None else x for x in adslab.site_properties["bulk_equivalent"]],
    )
    adslab.add_site_property(
        "surface_properties",
        [
            "adsorbate" if x is None else x
            for x in adslab.site_properties["surface_properties"]
        ],
    )

    return adslab


# Try the clockwise thing again...
def get_clockwise_rotations(slab_ref, slab, molecule):
    """We need to rush function..."""
    # This will be a inner method
    mxidegen = MXideAdsorbateGenerator(
        slab_ref,
        repeat=[1, 1, 1],
        verbose=False,
        positions=["MX_adsites"],
        # relax_tol=0.025,
        # tol=1.2,
    )

    # Getting the bulk-like adsites on the original slab
    bulk_like, _ = mxidegen.get_bulk_like_adsites()
    # bulk_like_sites = mxidegen._filter_clashed_sites(bulk_like)  # is needed?

    # Bondlength and X
    _, X = mxidegen.bondlengths_dict, mxidegen.X
    bulk_like_shifted = _bulk_like_adsites_perturbation(slab_ref, slab, bulk_like, X=X)

    # set n_rotations to 1 if mono-atomic
    n = len(molecule[0]) if type(molecule).__name__ == "list" else len(molecule)
    n_rotations = 1 if n == 1 else 4

    # Angles
    angles = get_angles(n_rotations=n_rotations)

    # Molecule formula
    molecule_comp = molecule.composition.as_dict()
    molecule_formula = "".join(molecule_comp.keys())

    # rotate OH
    molecule_rotations = mxidegen.get_transformed_molecule_MXides(
        molecule, axis=[0, 0, 1], angles_list=angles
    )

    # random_index = np.random.randint(len(bulk_like_shifted) - 1)
    # bulk_like_shifted = [bulk_like_shifted[random_index]]
    # placement
    adslab_dict = {}
    for rot_idx in range(len(molecule_rotations)):
        slab_ads = slab.copy()
        slab_ads = add_adsorbates(
            slab_ads, bulk_like_shifted, molecule_rotations[rot_idx]
        )
        slab_ads.sort()
        adslab_dict.update({"{}_{}".format(molecule_formula, rot_idx + 1): slab_ads})

    return adslab_dict, bulk_like_shifted


def _bulk_like_adsites_perturbation(slab_ref, slab, bulk_like_sites, X):
    """Let's perturb bulk_like_sites with delta (x,y,z) comparing input and output"""
    slab_ref_coords = slab_ref.cart_coords
    slab_coords = slab.cart_coords

    delta_coords = slab_coords - slab_ref_coords

    metal_idx = []
    for bulk_like_site in bulk_like_sites:
        min_dist = np.inf  # initialize min_dist register
        min_metal_idx = 0  # initialize min_metal_idx
        end_idx = np.where(slab_ref.frac_coords[:, 2] >= slab_ref.center_of_mass[2])[0][
            -1
        ]
        for idx, site in enumerate(
            slab_ref
        ):  # FIXME: I think we can make this faster by replacing with a while loop
            # and only looping over the top half
            if (
                site.specie != Element(X)
                and site.frac_coords[2]
                > slab_ref.center_of_mass[2]  # go over the top half of slab
            ):
                dist = np.linalg.norm(bulk_like_site - site.coords)
                if dist < min_dist:
                    min_dist = dist
                    min_metal_idx = idx
            if idx == end_idx:  # make sure that len(bulk_like_sites) == len(metal_idx)
                metal_idx.append(min_metal_idx)

    bulk_like_deltas = [delta_coords[i] for i in metal_idx]
    return [n + m for n, m in zip(bulk_like_sites, bulk_like_deltas)]


def SurfacePBX_WF(
    bulk_structure,
    slab,
    slab_orig,
    slab_uuid,
    oriented_uuid,
    adsorbates,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
    metal_site="",
    applied_potential=1.6,
    applied_pH=0,
    # streamline=False,
    checkpoint_path=None,
    is_metal=False,
):
    """
    Wrap-up Workflow for surface-OH/Ox terminated + SurfacePBX Analysis.

    Args:

    Retruns:
        something
    """
    # Empty list of fws
    hkl_fws, hkl_uuids = [], []

    # Reduced formula and Miller_index
    reduced_formula = slab.composition.reduced_formula
    slab_miller_index = "".join(list(map(str, slab.miller_index)))

    # Generate a set of OptimizeFW additons that will relax all the adslab in parallel
    ads_slab_orig = {}
    adslabs = {}
    for adsorbate in adsorbates:
        if (
            not checkpoint_path or len(adsorbate) == 1
        ) and not is_metal:  # As before # FIXME: Make sure this works
            adslab, bulk_like_shifted = get_clockwise_rotations(
                slab_orig, slab, adsorbate
            )
            adslabs.update(adslab)
        else:
            # Find the most stable config with adsorbate monolayer
            surface_pbx_ml = SurfaceCoverageML(
                slab_orig, slab, adsorbate, is_metal, checkpoint_path=checkpoint_path
            )
            adslabs.update(
                {
                    f"{surface_pbx_ml.adsorbate_formula}_1": surface_pbx_ml.pmg_stable_config
                }
            )
            bulk_like_shifted = surface_pbx_ml.bulk_like_shifted
    for adslab_label, adslab in adslabs.items():
        name = f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
        ads_slab_uuid = str(uuid.uuid4())
        adslab.remove_oxidation_states()  # for serialization purposes
        ads_slab_fw = AdsSlab_FW(
            adslab,
            name=name,
            oriented_uuid=oriented_uuid,
            slab_uuid=slab_uuid,
            ads_slab_uuid=ads_slab_uuid,
            vasp_cmd=vasp_cmd,
            db_file=DB_FILE,
            run_fake=run_fake,
        )
        ads_slab_orig.update({adslab_label: adslab})
        hkl_fws.append(ads_slab_fw)
        hkl_uuids.append(ads_slab_uuid)

    # Surface PBX Diagram for each surface orientation
    surface_pbx_uuid = str(uuid.uuid4())
    pbx_name = f"Surface-PBX-{slab.composition.reduced_formula}-{slab_miller_index}"
    pbx_fw = SurfacePBX_FW(
        reduced_formula=reduced_formula,
        name=pbx_name,
        miller_index=slab_miller_index,
        slab_uuid=slab_uuid,
        oriented_uuid=oriented_uuid,
        ads_slab_uuids=hkl_uuids,
        parents=hkl_fws,
        db_file=DB_FILE,
        run_fake=run_fake,
        surface_pbx_uuid=surface_pbx_uuid,
    )

    oer_fw = OER_WF(
        bulk_structure=bulk_structure,
        miller_index=slab_miller_index,
        slab_orig=slab_orig,
        bulk_like_sites=bulk_like_shifted,
        ads_dict_orig=ads_slab_orig,
        metal_site=metal_site,
        applied_potential=applied_potential,
        applied_pH=applied_pH,
        parents=[pbx_fw],
        run_fake=run_fake,
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
        surface_pbx_uuid=surface_pbx_uuid,
        # streamline=streamline,
        checkpoint_path=">>checkpoint_path<<",
    )

    all_fws = hkl_fws + [pbx_fw] + [oer_fw]
    oer_wf = Workflow(
        all_fws,
        name=f"{slab.composition.reduced_formula}-{slab_miller_index}-PBX Workflow",
    )
    # breakpoint()
    return oer_wf
