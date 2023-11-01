from fireworks import (
    Firework,
    Workflow,
    FWAction,
    explicit_serialize,
    FiretaskBase,
    LaunchPad,
)
from atomate.vasp.database import VaspCalcDb
from atomate.utils.utils import env_chk
import os
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from ase.optimize import LBFGS
import numpy as np
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import torch
import re

# from ocpmodels.common.relaxation.ase_utils import OCPCalculator


@explicit_serialize
class ML_int_relax(FiretaskBase):

    required_params = [
        "label",
        "uuid",
        "is_bulk",
        "template_ckpt",
        "finetune_ckpt",
        "structure",
        "db_file",
    ]

    def run_task(self, fw_spec):

        label = self["label"]
        uuid = self["uuid"]
        is_bulk = self["is_bulk"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        # template_ckpt = env_chk(self["template_ckpt"], fw_spec)
        finetune_ckpt = env_chk(self["finetune_ckpt"], fw_spec)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        # old_cp = torch.load(template_ckpt, map_location=device)
        # new_cp = torch.load(finetune_ckpt, map_location=device)
        # We go over the old checkpoint and make the necessary updates
        # for param_name in old_cp["state_dict"]:
        #    try:
        #        # print(new_cp["state_dict"][param_name] - old_cp["state_dict"][param_name])
        #        old_cp["state_dict"][param_name] = new_cp["state_dict"][
        #            "backbone" + param_name.split("module.module")[1]
        #        ]
        #    except KeyError:
        #        old_cp["state_dict"][param_name] = new_cp["state_dict"][param_name]

        # old_cp["config"]["model_attributes"]["qint_tags"] = [0, 1, 2]
        # torch.save(
        #    old_cp,
        #    "/home/jovyan/makesureINFfinetune_NRC_data_MAEwith_scaling-epoch=80-step=648-val_loss=0.1440.ckpt",
        # )
        # Here we load the checkpoint with the finetuned weights into an OCPCalculator
        ocp_calculator = OCPCalculator(checkpoint_path=finetune_ckpt)
        # We go over the old checkpoint and make the necessary updates
        structure = self["structure"]
        # Convert the serializable structure back to ASE for the relaxation
        atoms = AAA.get_atoms(structure)
        # if is_bulk:
        #    tags = np.array([0] * len(structure))
        #    # Check the constraints
        # elif "constraints" in atoms.todict():
        #    tags = np.array(
        #        [
        #            0
        #            if atom.index in atoms.todict()["constraints"][0].get_indices()
        #            else 1
        #            for atom in atoms
        #        ]
        #    )
        #    # TODO Logic for adsorbate tagging
        # elif "constraints" not in atoms.todict() and not fw_spec.get(
        #    "is_adslab"
        # ):  # slab
        #    tags = np.array([1] * len(structure))
        # if "_" in label:
        #    split_key = label.split("_")[0]
        # else:
        #    split_key = label
        # if split_key == "OOH":
        #    tags[np.array([0, 51, 52, 53, 54])] = 2
        # elif split_key == "OH":
        #    tags[np.array([0, 51, 52, 53])] = 2
        # elif split_key == "Ox":
        #    tags[np.array([50, 51, 52])] = 2
        # else:  # Clean
        #    tags[np.array([50, 51])] = 2
        os.makedirs("data", exist_ok=True)
        atoms.set_calculator(ocp_calculator)
        atoms.set_pbc(True)
        if is_bulk:  # Bulk
            atoms.set_tags(
                [0] * len(structure)
            )  # This needs to be set to 1 if using the pre-trained model since they did not trained on tagged bulks
        elif not is_bulk and not fw_spec.get("is_adslab"):  # Slab
            if "constraints" in atoms.todict().keys():
                tags = np.array(
                    [
                        0
                        if atom.index in atoms.todict()["constraints"][0].get_indices()
                        else 1
                        for atom in atoms
                    ]
                )
            else:
                tags = np.array([1] * len(structure))
            atoms.set_tags(tags)
        else:  # Adslab
            atoms.set_tags(atoms.todict()["tags"])
        # Make sure the tags don't come out as object dtype
        atoms.arrays["tags"] = atoms.arrays["tags"].astype(int)
        orig_structure = AAA.get_structure(atoms).copy()
        dyn = LBFGS(atoms, trajectory=f"data/{label}.traj")
        dyn.run(fmax=0.03, steps=100)
        relaxed_energy = atoms.get_potential_energy()
        # We then relax the atomic structure and update_spec with the relaxed energy for the analysis
        # firetask
        relaxed_structure = AAA.get_structure(atoms)
        relaxed_structure.remove_oxidation_states()
        orig_structure.remove_oxidation_states()
        # Insert the results into the task collection per the atomate schema and using the uuid
        task_doc = {
            "uuid": uuid,
            "calcs_reversed": [
                {
                    "output": {
                        "structure": None,
                        "energy": 0,
                        "ionic_steps": [
                            {"e_0_energy": 0, "structure": None},
                            {"e_0_energy": 0, "structure": None},
                        ],
                    }
                }
            ],
        }
        relaxed_forces = atoms.get_forces().tolist()
        task_doc["calcs_reversed"][0]["output"][
            "structure"
        ] = relaxed_structure.as_dict()
        task_doc["calcs_reversed"][0]["output"]["energy"] = relaxed_energy
        task_doc["calcs_reversed"][0]["output"]["ionic_steps"][0][
            "structure"
        ] = orig_structure.as_dict()
        task_doc["calcs_reversed"][0]["output"]["ionic_steps"][-1][
            "structure"
        ] = relaxed_structure.as_dict()
        task_doc["calcs_reversed"][0]["output"]["ionic_steps"][-1][
            "e_0_energy"
        ] = relaxed_energy
        task_doc["calcs_reversed"][0]["output"]["forces"] = relaxed_forces
        if not is_bulk:
            task_doc.update(
                {
                    "slab": fw_spec["slab"].as_dict(),
                    "parent_structure": fw_spec["parent_structure"].as_dict(),
                    "parent_structure_metadata": fw_spec["parent_structure_metadata"],
                }
            )
        mmdb.db["tasks"].insert_one(task_doc)

        return FWAction(
            #    update_spec={
            #        f"{label}_relaxed_energy": relaxed_energy,
            #        f"{label}_relaxed_structure": relaxed_structure,
            #        f"{label}_orig_structure": orig_structure,
            #    }
        )


@explicit_serialize
class analyze_ML_OER_results(FiretaskBase):
    def run_task(self, fw_spec):
        # Partition the energies for the ones with degrees of freedom
        ooh_dict = {k: v for k, v in fw_spec.items() if re.search("^OOH_.*energy", k)}
        ooh_orig_structs = {
            k: v for k, v in fw_spec.items() if re.search("^OOH_.*orig_structure", k)
        }
        ooh_relaxed_structs = {
            k: v for k, v in fw_spec.items() if re.search("^OOH_.*relaxed_structure", k)
        }
        non_deprotonated_ooh = {}
        for orig_key, orig_struct in ooh_orig_structs.items():
            # Locate the indices from the original structure
            orig_struct.sort()
            search_r = 1
            # Find the hydrogen index
            h_index = np.where(
                np.array([site.species_string for site in orig_struct]) == "H"
            )[0].item()
            h_site = orig_struct[h_index]
            while (
                len(fw_spec[orig_key].get_sites_in_sphere(h_site.coords, search_r)) < 2
            ):
                search_r += 0.01
            new_sites = []
            # Check if there are any sites that are subject to PBC
            for site in fw_spec[orig_key].get_sites_in_sphere(h_site.coords, search_r):
                if site.species_string != "H":
                    site.frac_coords[
                        np.where(
                            (np.round(site.frac_coords, 2) >= 1)
                            | (np.round(site.frac_coords, 2) < 0)
                        )
                    ] = np.mod(
                        site.frac_coords[
                            (np.round(site.frac_coords, 2) >= 1)
                            | (np.round(site.frac_coords, 2) < 0)
                        ],
                        1,
                    )
                new_sites.append(site)
            ooh_indices = [orig_struct.index(s) for s in new_sites]
            relaxed_struct = ooh_relaxed_structs[
                orig_key.split("orig_structure")[0] + "relaxed_structure"
            ]
            relaxed_struct.sort()
            # Check the pairwise bond distance between H and O (should be less than or equal to 1.1A)
            HO_distance = relaxed_struct[h_index].distance(
                relaxed_struct[[i for i in ooh_indices if i != h_index][0]]
            )
            print("OOH_check", HO_distance, orig_key)
            if HO_distance < 1.1:
                non_deprotonated_ooh[
                    orig_key.split("orig_structure")[0] + "relaxed_energy"
                ] = fw_spec[orig_key.split("orig_structure")[0] + "relaxed_energy"]
        # FIXME: Need to add logic for checking whether the OOH is de-protonated or not and exclude those candidates
        # before taking the minimum energy
        oh_dict = {
            k: v for k, v in fw_spec.items() if re.search("^OH_.*relaxed_energy", k)
        }
        oh_orig_structs = {
            k: v for k, v in fw_spec.items() if re.search("^OH_.*orig_structure", k)
        }
        oh_relaxed_structs = {
            k: v for k, v in fw_spec.items() if re.search("^OH_.*relaxed_structure", k)
        }
        non_deprotonated_oh = {}
        for orig_key, orig_struct in oh_orig_structs.items():
            try:
                # Locate the indices from the original structure
                orig_struct.sort()
                search_r = 1
                # Find the hydrogen index
                h_index = np.where(
                    np.array([site.species_string for site in orig_struct]) == "H"
                )[0].item()
                h_site = orig_struct[h_index]
                while (
                    len(fw_spec[orig_key].get_sites_in_sphere(h_site.coords, search_r))
                    < 2
                ):
                    search_r += 0.01
                new_sites = []
                # Check if there are any sites that are subject to PBC
                for site in fw_spec[orig_key].get_sites_in_sphere(
                    h_site.coords, search_r
                ):
                    if site.species_string != "H":
                        site.frac_coords[
                            np.where(
                                (np.round(site.frac_coords, 2) >= 1)
                                | (np.round(site.frac_coords, 2) < 0)
                            )
                        ] = np.mod(
                            site.frac_coords[
                                (np.round(site.frac_coords, 2) >= 1)
                                | (np.round(site.frac_coords, 2) < 0)
                            ],
                            1,
                        )
                    new_sites.append(site)
                oh_indices = [orig_struct.index(s) for s in new_sites]
                relaxed_struct = oh_relaxed_structs[
                    orig_key.split("orig_structure")[0] + "relaxed_structure"
                ]
                relaxed_struct.sort()
                # Check the pairwise bond distance between H and O (should be less than or equal to 1.1A)
                HO_distance = relaxed_struct[h_index].distance(
                    relaxed_struct[[i for i in oh_indices if i != h_index][0]]
                )
                print("OH_check", HO_distance, orig_key)
                if HO_distance < 1.1:
                    non_deprotonated_oh[
                        orig_key.split("orig_structure")[0] + "relaxed_energy"
                    ] = fw_spec[orig_key.split("orig_structure")[0] + "relaxed_energy"]
            except:
                continue

        # Get the lowest energy configuration
        min_ooh_key = min(non_deprotonated_ooh, key=non_deprotonated_ooh.get)
        min_oh_key = min(non_deprotonated_oh, key=non_deprotonated_oh.get)
        E_ref = fw_spec["Clean_relaxed_energy"]
        E_OH = oh_dict[min_oh_key] - E_ref - (-14.25994015 - (-0.5 * 6.77818501)) + 0.36
        E_OOH = (
            non_deprotonated_ooh[min_ooh_key]
            - E_ref
            - (2 * (-14.25994015) - 1.5 * (-6.77818501))
            + 0.44
        )
        E_Ox = (
            fw_spec["Ox_relaxed_energy"] - E_ref - (-14.25994015 - -6.77818501) + 0.08
        )
        G_Ox_OH = E_Ox - E_OH
        G_OOH_Ox = E_OOH - E_Ox
        GO2 = 4.92 - E_OOH
        G_OH = E_OH
        overpotential = max(G_OH, GO2, G_OOH_Ox, G_Ox_OH) - 1.23

        return FWAction(
            stored_data={
                "G_Ox": E_Ox,
                "G_OOH": E_OOH,
                "G_OH": E_OH,
                "overpotential": overpotential,
                "G_Ox_OH": E_Ox - E_OH,
                "G_OOH_Ox": E_OOH - E_Ox,
                "GO2": 4.92 - E_OOH,
            }
        )


# depth_1_dirs = next(os.walk("./"))[1]
# launchpad = LaunchPad(
#    host="localhost",
#    name="fw_oal",
#    port=27017,
#    username="fw_oal_admin",
#    password="gfde223223222rft3",
# )
# parents = []
# fws = []
# for dir_ in depth_1_dirs:
#    atoms = read(f"{dir_}/POSCAR")
#    struct = AAA.get_structure(atoms)
#    fw = Firework(
#        ML_int_relax(
#            label=f"{dir_}",
#            structure=struct,
#            finetune_ckpt=">>finetune_ckpt<<",
#            template_ckpt=">>template_ckpt<<",
#        ),
#        name=f"{dir_}",
#    )
#    breakpoint()
#    parents.append(fw)
#    fws.append(fw)
# analysis_fw = Firework(
#    analyze_ML_OER_results(test="test"), name="OER_int_analysis", parents=parents
# )
# fws.append(analysis_fw)
# wf = Workflow(fws, name="ML_int_relax_wf_Mo_Nb")
# launchpad.add_wf(wf)
