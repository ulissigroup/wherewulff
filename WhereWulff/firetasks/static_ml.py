"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import uuid
import json
import itertools

import numpy as np
from scpipy.spatial.distance import pdist, squareform
from pydash.objects import has, get

# ASE/Pymatgen
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

# OCP
import torch
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.preprocessing import AtomsToGraph
from ocpmodels.datasets import data_list_collater
from ocpmodels.common.utils import setup_imports
from ocpmodels.trainers import ForcesTrainer

# WhereWulff

# Fireworks/Atomate
from fireworks import FireTaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.config import VASP_CMD, DB_FILE

logger = get_logger(__name__)

@explicit_serialize
class StaticMLFireTask(FireTaskBase):
    """
    Given a structure predict the DFT energy,
    using OCP models.

    Args:
        structure        (pmg struct): Pymatgen Structure object, usually slab or ads_slab structure.
        model_checkpoint (path)      : Path to the best OCP model checkpoint.
        model_config     (path)      : Path to the OCP model configuration file.
        to_db            (bool)      : If results should be stored into db or local json file.
        db_file          (env)       : Environment variable to connect to the DB.

    returns:
        StaticML FireTask.
    """

    required_params = ["structure", "model_checkpoint", "model_config", "db_file"]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Global Variables
        structure = self["structure"]
        model_checkpoint = self["model_checkpoint"]
        model_config = self["model_config"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        summary_dict = {"structure": structure,
                        "model_checkpoint": model_checkpoint,
                        "model_config": model_config}

        # Init OCP calculator
        calc = OCPCalculator(model_config, 
                             checkpoint=model_checkpoint)

        # Init Atoms2Graph 
        a2g = AtomsToGraph(max_neigh=50,
                           radius=6,
                           r_energy=False,
                           r_forces=False,
                           r_distance=False,
                           r_edges=False,)

        # Convert PMG structure to ASE
        atoms = AAA.get_atoms(structure)

        # Apply the tags for surface/adsorbate
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
        atoms.set_tags(tags)

        # Convert to graph
        graph = a2g.convert(atoms)

        # Place the tgas on the graph object
        graph.tags = torch.LongTensor(atoms.get_tags().astype(int))

        # Given the graph, predict
        batch = data_list_collater([graph], otf_graph=True)
        prediction = calc.trainer.predict(batch, per_image=False, disable_tqdm=True)

        # Store structure and prediction
        summary_dict["prediction"] = prediction

        # Logger
        logger.info(f"{structure.composition.reduced_formula} predicted DFT energy is: {prediction} eV")
        
        return