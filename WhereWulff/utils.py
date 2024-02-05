from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.preprocessing import AtomsToGraphs
from pymatgen.io.ase import AseAtomsAdaptor as AAA
import torch
from ocpmodels.datasets import data_list_collater


def find_most_stable_config(configs, checkpoint_path):
    # Init the OCP calculator
    calc = OCPCalculator(checkpoint_path=checkpoint_path)

    # Init Atoms2Graph
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=False,
    )
    adslab_atoms = []
    for config in configs:
        atoms = AAA.get_atoms(config)
        # Apply the tags for relaying to gemnet model
        if "constraints" not in atoms.todict():
            constraint_indices = [1] * len(atoms)
        else:
            constraint_indices = atoms.todict()["constraints"][
                0
            ].get_indices()  # NOTE: This seamless transfer of properties seems to be only
            # present in later versions of pymatgen
        surface_properties = atoms.todict()["surface_properties"] == "adsorbate"
        tags = []
        for (is_adsorbate, atom) in zip(surface_properties, atoms):
            if atom.index in constraint_indices:
                tags.append(0)  # bulk like
            elif atom.index not in constraint_indices and not is_adsorbate:
                tags.append(1)  # surface
            else:
                tags.append(2)  # adsorbate
        # breakpoint()
        atoms.set_tags(tags)
        # Make sure the tags array does not come out as a dtype object
        atoms.arrays["tags"] = atoms.arrays["tags"].astype(int)
        adslab_atoms.append(atoms)
    graphs = a2g.convert_all(adslab_atoms)

    # Place the tags on the graph objects
    for graph, atoms in zip(graphs, adslab_atoms):
        graph.tags = torch.LongTensor(
            atoms.get_tags().astype(int)
        )  # Need the dtype to be LongTensor and items to be int type

    batch = data_list_collater(graphs, otf_graph=True)
    predictions = calc.trainer.predict(batch, per_image=False, disable_tqdm=True)[
            "energy"
            ]
    breakpoint()
    if len(predictions.shape) >= 2:
        predictions = predictions.squeeze(1)
        stable_index = predictions.sort().indices[0]
    slab_ads = AAA.get_structure(adslab_atoms[stable_index])
    return slab_ads, stable_index
