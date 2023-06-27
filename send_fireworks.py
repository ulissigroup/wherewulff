from fireworks import (
    Firework,
    Workflow,
    FWAction,
    explicit_serialize,
    FiretaskBase,
    LaunchPad,
)
from atomate.utils.utils import env_chk
import os
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from ase.optimize import LBFGS
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import torch

# from ocpmodels.common.relaxation.ase_utils import OCPCalculator


@explicit_serialize
class ML_int_relax(FiretaskBase):

    required_params = ["template_ckpt", "finetune_ckpt", "structure"]

    def run_task(self, fw_spec):

        template_ckpt = env_chk(self["template_ckpt"], fw_spec)
        finetune_ckpt = env_chk(self["finetune_ckpt"], fw_spec)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        old_cp = torch.load(template_ckpt, map_location=device)
        new_cp = torch.load(finetune_ckpt, map_location=device)
        # We go over the old checkpoint and make the necessary updates
        for param_name in old_cp["state_dict"]:
            try:
                # print(new_cp["state_dict"][param_name] - old_cp["state_dict"][param_name])
                old_cp["state_dict"][param_name] = new_cp["state_dict"][
                    "backbone" + param_name.split("module.module")[1]
                ]
            except KeyError:
                old_cp["state_dict"][param_name] = new_cp["state_dict"][param_name]

        old_cp["config"]["model_attributes"]["qint_tags"] = [0, 1, 2]
        torch.save(
            old_cp,
            "/home/jovyan/makesureINFfinetune_NRC_data_MAEwith_scaling-epoch=80-step=648-val_loss=0.1440.ckpt",
        )
        breakpoint()
        ocp_calculator = OCPCalculator(checkpoint=finetune_ckpt)
        # We go over the old checkpoint and make the necessary updates
        structure = self["structure"]
        # Convert the serializable structure back to ASE for the relaxation
        atoms = AAA.get_atoms(structure)
        # Here we load the checkpoint with the finetuned weights into an OCPCalculator

        # We then relax the atomic structure and update_spec with the relaxed energy for the analysis
        # firetask

        return FWAction(update_spec={})


@explicit_serialize
class analyze_ML_OER_results(FiretaskBase):
    required_params = ["test"]

    def run_task(self, fw_spec):
        test = self["test"]

        return FWAction()


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
#            structure=struct,
#            finetune_ckpt=">>finetune_ckpt<<",
#            template_ckpt=">>template_ckpt<<",
#        ),
#        name=f"{dir_}",
#    )
#    parents.append(fw)
#    fws.append(fw)
# analysis_fw = Firework(
#    analyze_ML_OER_results(test="test"), name="OER_int_analysis", parents=parents
# )
# fws.append(analysis_fw)
# wf = Workflow(fws, name="ML_int_relax_wf_Mo_Nb")
# launchpad.add_wf(wf)
