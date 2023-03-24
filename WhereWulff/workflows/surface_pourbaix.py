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

    def __init__(self, slab_ref, slab, adsorbate, checkpoint_path):

        self.checkpoint_path = checkpoint_path
        # Init Mxide
        mxidegen = MXideAdsorbateGenerator(
            slab_ref, repeat=[1, 1, 1], verbose=False, positions=["MX_adsites"]
        )

        self.bulk_like, _ = mxidegen.get_bulk_like_adsites()
        _, self.X = mxidegen.bondlengths_dict, mxidegen.X

        # Perturb bulk_like_sites in case the slab model is optimized
        self.bulk_like_shifted = self._bulk_like_adsites_perturbation(
            slab_ref=slab_ref, slab=slab
        )

        # Set n_rotations
        n = len(adsorbate[0]) if type(adsorbate).__name__ == "list" else len(adsorbate)
        n_rotations = 1 if n == 1 else 8

        # Get angles
        self.angles = self._get_angles(n_rotations=n_rotations)

        # Adsorbate formula
        adsorbate_comp = adsorbate.composition.as_dict()
        self.adsorbate_formula = "".join(adsorbate_comp.keys())

        # Rotate adsorbate
        self.adsorbate_rotations = mxidegen.get_transformed_molecule_MXides(
            adsorbate, axis=[0, 0, 1], angles_list=self.angles
        )

        # (pseudo)-Randomly pick the first site
        site_index = np.random.choice(len(self.bulk_like_shifted))
        remaining_site_indices = list(range(len(self.bulk_like_shifted)))

        # Place the 1st adsorbate on that site
        square_distance_matrix = self._get_closest_neighbors(self.bulk_like_shifted)[1]

        # Loop
        counter = 0
        while remaining_site_indices:
            if len(remaining_site_indices) == len(self.bulk_like_shifted):
                slab_ads = slab.copy()

            (
                site_1,
                site_2,
                remaining_site_indices,
                square_distance_matrix,
            ) = self.find_and_update_sites(
                square_distance_matrix, remaining_site_indices
            )
            slab_ads = add_adsorbates(
                slab_ads.copy(),
                [self.bulk_like_shifted[site_1], self.bulk_like_shifted[site_2]],
                self.adsorbate_rotations[0],
            )
            configs = self.rotate_site_indices(slab_ads, counter)
            slab_ads = find_most_stable_config(
                configs, checkpoint_path=self.checkpoint_path
            )[0]
            counter += 1
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

    def find_and_update_sites(self, square_distance_matrix, remaining_site_indices):
        row, column = np.unravel_index(
            square_distance_matrix.argmin(), square_distance_matrix.shape
        )
        for index in [row, column]:
            remaining_site_indices.remove(index)
        square_distance_matrix[row, :] = np.inf
        square_distance_matrix[:, column] = np.inf
        square_distance_matrix[column, :] = np.inf
        square_distance_matrix[:, row] = np.inf
        return (row, column, remaining_site_indices, square_distance_matrix)

    def rotate_site_indices(self, slab_ads, counter):
        anchor_site_indices = np.where(
            (np.array(slab_ads.site_properties["binding_site"]) == True)
        )[0].tolist()[-2:]
        adsorbate_indices_two = np.where(
            (np.array(slab_ads.site_properties["surface_properties"]) == "adsorbate")
        )[0].tolist()[-len(self.adsorbate_rotations[0]) :]
        adsorbate_indices_one = np.where(
            (np.array(slab_ads.site_properties["surface_properties"]) == "adsorbate")
        )[0].tolist()[
            -2 * len(self.adsorbate_rotations[0]) : -len(self.adsorbate_rotations[0])
        ]

        configs = []
        # for site, other_site in rotate_site_indices:
        for i, ang in enumerate(self.angles):
            if len(configs) > 1:
                slab_ads = configs[len(configs) - len(self.angles)].copy()
            first_site = anchor_site_indices[0]
            slab_ads.rotate_sites(
                adsorbate_indices_one,
                ang,
                [0, 0, 1],
                slab_ads[first_site].coords,
                to_unit_cell=False,
            )
            # configs.append(slab_ads.copy())
            second_site = anchor_site_indices[1]
            for ang2 in self.angles:
                slab_ads.rotate_sites(
                    adsorbate_indices_two,
                    ang2,
                    [0, 0, 1],
                    slab_ads[second_site].coords,
                    to_unit_cell=False,
                )
                # slab_ads.to(filename=f"visuals/POSCAR_dimer_{counter}_{i}_{ang}_{ang2}")
                configs.append(slab_ads.copy())

        return configs

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
                    site.specie != Element(self.X)
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
        relax_tol=0.025,
        tol=1.2,
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
        if not checkpoint_path or len(adsorbate) == 1:
            adslabs, bulk_like_shifted = get_clockwise_rotations(
                slab_orig, slab, adsorbate
            )
        else:
            # TODO: Find the most stable config with adsorbate monolayer
            surface_pbx_ml = SurfaceCoverageML(
                slab_orig, slab, adsorbate, checkpoint_path=checkpoint_path
            )
            adslabs.update(
                {
                    f"{surface_pbx_ml.adsorbate_formula}_1": surface_pbx_ml.pmg_stable_config
                }
            )
            bulk_like_shifted = surface_pbx_ml.bulk_like_shifted
        for adslab_label, adslab in adslabs.items():
            name = (
                f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
            )
            ads_slab_uuid = str(uuid.uuid4())
            ads_slab_fw = AdsSlab_FW(
                adslab,
                name=name,
                oriented_uuid=oriented_uuid,
                slab_uuid=slab_uuid,
                ads_slab_uuid=ads_slab_uuid,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
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
        checkpoint_path=checkpoint_path,
    )

    all_fws = hkl_fws + [pbx_fw] + [oer_fw]
    oer_wf = Workflow(
        all_fws,
        name=f"{slab.composition.reduced_formula}-{slab_miller_index}-PBX Workflow",
    )
    return oer_wf
