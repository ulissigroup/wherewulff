"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import uuid
import json
import itertools

import numpy as np
from scipy.spatial.distance import pdist, squareform
from pydash.objects import has, get

# ASE/Pymatgen
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

# OCP
import torch
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.common.utils import setup_imports
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import data_list_collater
from ocpmodels.common.utils import setup_imports
from ocpmodels.trainers import ForcesTrainer

# WhereWulff
from WhereWulff.adsorption.MXide_adsorption import MXideAdsorbateGenerator

# Fireworks/Atomate
from fireworks import FireTaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)

@explicit_serialize
class SurfaceCoverageMLFireTask(FireTaskBase):
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

    required_params = ["slab", 
    "slab_ref", 
    "adsorbate",
    "miller_index",
    "slab_uuid",
    "oriented_uuid",
    "ads_slab_uuids",
    "model_checkpoint", 
    "model_config", 
    "db_file"
    ]
    optional_params = []

    def run_task(self, fw_spec):

        # Global variables
        slab = self["slab"]
        slab_ref = self["slab_ref"]
        adsorbate = self["adsorbate"]
        miller_index = self["miller_index"]
        slab_uuid = self["slab_uuid"]
        oriented_uuids = self["oriented_uuids"]
        ads_slab_uuids = self["ads_slab_uuids"]
        model_checkpoint = env_chk(self["model_checkpoint"], fw_spec)
        model_config = env_chk(self["model_config"], fw_spec)
        db_file = env_chk(self.get("db_file"), fw_spec)
        summary_dict = {"slab": slab.as_dict(),
                        "slab_ref": slab_ref.as_dict(),
                        "adsorbate": adsorbate.as_dict(),
                        "miller_index": miller_index,
                        "slab_uuid": slab_uuid,
                        "oriented_uuids": oriented_uuids,
                        "ads_slab_uuids": ads_slab_uuids,
                        "model_checkpoint": model_checkpoint,
                        "model_config": model_config,
        }
        # Seed for randomness
        np.random.seed(42)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Init the OCP calculator
        calc = OCPCalculator(model_config,
                    checkpoint=model_checkpoint)

        # Init Atoms2Graph
        a2g = AtomsToGraphs(max_neigh=50,
                            radius=6,
                            r_energy=False,
                            r_forces=False,
                            r_distances=False,
                            r_edges=False)

        # Init Mxide
        mxidegen = MXideAdsorbateGenerator(slab_ref,
                                          repeat=[1,1,1],
                                          verbose=False,
                                          positions=["MX_adsites"])

        self.bulk_like, _ = mxidegen.get_bulk_like_adsites()
        _, self.X = mxidegen.bondlengths_dict, mxidegen.X

        # Perturb bulk_like_sites in case the slab model is optimized
        self.bulk_like_shifted = self._bulk_like_adsites_perturbation(slab_ref=slab_ref, slab=slab)

        # Set n_rotations
        n = len(adsorbate[0]) if type(adsorbate).__name__ == "list" else len(adsorbate)
        n_rotations = 1 if n == 1 else 4

        # Get angles
        angles = self._get_angles(n_rotations=n_rotations)

        # Adsorbate formula
        adsorbate_comp = adsorbate.composition.as_dict()
        adsorbate_formula = "".join(adsorbate_comp.keys())

        # Rotate adsorbate
        adsorbate_rotations = mxidegen.get_transformed_molecule_MXides(
            adsorbate, axis=[0, 0, 1], angles_list=angles
        )

        # (pseudo)-Randomly pick the first site
        site_index = np.random.choice(len(self.bulk_like_shifted))
        remaining_site_indices = list(range(len(self.bulk_like_shifted)))

        # Place the 1st adsorbate on that site
        square_distance_matrix = self._get_closest_neighbors(self.bulk_like_shifted)[1]

        # Loop
        while remaining_site_indices:
            adslab_atoms = []
            for i, mol in enumerate(adsorbate_rotations):
                if len(remaining_site_indices) == len(self.bulk_like_shifted):
                    slab_ads_orig = slab.copy()
                elif i == 0:
                    slab_ads_orig = slab_ads.copy()
                slab_ads = self._add_adsorbate(
                    slab_ads_orig.copy(), [self.bulk_like_shifted[site_index]], mol
                )

                # Get ASE-object with tags
                atoms, tags = self._get_ase_object(struct=slab_ads)
                adslab_atoms.append(atoms)

            # From atoms to graphs
            graphs = a2g.convert_all(adslab_atoms)

            # Place the tags on the graph objects
            for graph, atoms in zip(graphs, adslab_atoms):
                graph.tags = torch.LongTensor(
                    atoms.get_tags().astype(int)
                )

            # Batch of graphs ready to predict
            batch = data_list_collater(graphs, otf_graph=True)
            predictions = calc.trainer.predict(
                batch, per_image=False, disable_tqdm=True
            )

            # Sort the predictions to find the most stable configuration
            # with one site populated
            stable_config_index = predictions["energy"].sort().indices[0].item()

            stable_config = adslab_atoms[stable_config_index].copy()
            pmg_stable_config = AAA.get_structure(stable_config)

            # Set the next structure to be the most stable from this pass
            slab_ads = pmg_stable_config.copy()

            if len(remaining_site_indices) == len(self.bulk_like_shifted):
                previous_index = site_index
                site_index = square_distance_matrix[site_index].argmin()
                remaining_site_indices.remove(previous_index)

            # Now place the next adsorbate on the site that is closest to the previous one
            # Get index of closest site to site_index
            else:
                square_distance_matrix[:, previous_index] = np.inf
                previous_index = site_index
                site_index = square_distance_matrix[site_index].argmin()
                remaining_site_indices.remove(previous_index)

        # Store stable config into summary_dict
        summary_dict["bulk_like_shifted"] = self.bulk_like_shifted
        summary_dict["stable_config"] = pmg_stable_config.as_dict()
        pmg_stable_config.to(filename="POSCAR_most_stable")

        # Collection
        mmdb.collection = mmdb.db[f"{slab.composition.reduced_formula}_{adsorbate_formula} surface_coverage_ml"]
        mmdb.collection.insert_one(summary_dict)

        # Logger
        logger.info("Surface Coverage with ML, Done!")

        return

    def _get_angles(self, n_rotations=4):
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
        site_pairs = list(zip(range(len(self.bulk_like_shifted)), distances.argmin(axis=0)))
        return site_pairs, distances

    def _bulk_like_adsites_perturbation(self, slab_ref, slab):
        """Let's perturb bulk_like_sites with delta (x,y,z) comparing input and output"""
        slab_ref_coords = slab_ref.cart_coords
        slab_coords = slab.cart_coords

        delta_coords = slab_coords - slab_ref_coords

        metal_idx = []
        for bulk_like_site in self.bulk_like:
            min_dist = np.inf # initialize min_dist register
            min_metal_idx = 0
            end_idx = np.where(slab_ref.frac_coords[:, 2] >= slab_ref.center_of_mass[2])[0][
            -1
            ]
            for idx, site in enumerate(slab_ref):
                if (
                    site.specie != Element(self.X)
                    and site.frac_coords[2]
                    > slab_ref.center_of_mass[2]
                ):
                    dist = np.linalg.norm(bulk_like_site - site.coords)

                    if dist < min_dist:
                        min_dist = dist
                        min_metal_idx = idx

                if idx == end_idx:
                    metal_idx.append(min_metal_idx)

        bulk_like_deltas = [delta_coords[i] for i in metal_idx]
        return [n + m for n, m in zip(self.bulk_like, bulk_like_deltas)]

    def _add_adsorbate(self, adslab, ads_coords, molecule, z_offset=[0,0,0.15]):
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

    def _get_ase_object(self, struct):
        """Convert to ASE-object and add constraints"""
        # convert PMG structure to ASE
        atoms = AAA.get_atoms(struct)
        # Apply the tags for relaying to gemnet model
        constraint_indices = atoms.todict()["constraints"][0].get_indices()
        surface_properties = atoms.todict()["surface_properties"]
        tags = []
        for (is_adsorbate, atom) in zip(surface_properties, atoms):
            if atom.index in constraint_indices:
                tags.append(0) # bulk-like
            elif atom.index not in constraint_indices and not is_adsorbate:
                tags.append(1) # surface
            else:
                tags.append(2) # adsorbate
        atoms.set_tags(np.array(tags).astype(int))
        return atoms, tags
