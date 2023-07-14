import uuid
import numpy as np

from pydash.objects import has, get

from pymatgen.core import Structure
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.database import VaspCalcDb
from atomate.common.firetasks.glue_tasks import (
    CopyFiles,
    DeleteFilesPrevFolder,
    GzipDir,
)

from WhereWulff.dft_settings.settings import MOSurfaceSet
from WhereWulff.common.glue_tasks import GzipPrevDir
from WhereWulff.analysis.EDL import EDLAnalysis
from atomate.vasp.fireworks.core import StaticFW
from fireworks import Workflow, Firework


@explicit_serialize
class ContinueOptimizeFW(FiretaskBase):
    """
    Custom OptimizeFW Firetask that handles wall-time issues

    Args:
        is_bulk (bool): Determines DFT settings depending if its bulk or slab model.
        counter (int) : Counter wheter is a parent (counter = 0) or child job (counter > 0).
        db_file       : Environment variable check to be able to connect to the database.
        vasp_cmd      : Environment variable to execute vasp.

    Returns:
        Return a continuous Firetask that handles wall_times and files transfering
        between parent -> child -> terminal node.

    """

    required_params = ["is_bulk", "counter", "db_file", "vasp_cmd"]
    optional_params = []

    def run_EDL(self):
        # Logic to spawn a static set of single points with varying NELECT to simulate EDL and get free energy as fxn of applied potential on the SHE scale
        nelect_orig = self.task_doc["calcs_reversed"][0]["output"]["outcar"]["nelect"]
        contcar = self.task_doc["calcs_reversed"][0]["output"]["ionic_steps"][-1][
            "structure"
        ]
        # Add the LVHAR keyword and set NSW to zero
        incar_dict = self.fw_spec["_tasks"][0]["vasp_input_set"]
        orig_struct = Structure.from_dict(self.fw_spec["_tasks"][0]["structure"])
        orig_struct.sort()
        orig_magmoms = orig_struct.site_properties["magmom"]
        contcar_obj = Structure.from_dict(contcar)
        contcar_obj.add_site_property("magmom", orig_magmoms)
        contcar = contcar_obj.as_dict()
        incar_dict["user_incar_settings"] = {
            "NSW": 0,
            "LVHAR": True,
            "LSOL": True,
            "LAMBDA_D_K": 3.0,
        }
        incar_dict["structure"] = contcar
        parents = []
        all_fws = []
        uuids = []
        for nelect in np.arange(nelect_orig - 0.6, nelect_orig + 0.61, 0.2):  # SP FWs
            new_uuid = uuid.uuid4()
            uuids.append(new_uuid)
            incar_dict["user_incar_settings"].update({"NELECT": nelect})
            vasp_input_set = MOSurfaceSet.from_dict(incar_dict)
            static_fw = OptimizeFW(
                structure=Structure.from_dict(contcar),
                max_force_threshold=None,
                vasp_cmd=">>vasp_cmd<<",
                db_file=">>db_file<<",
                job_type="normal",
                name=f"SP_{nelect}",
                vasp_input_set=vasp_input_set,
            )
            static_fw.tasks[1].update({"gzip_output": False})
            static_fw.tasks[3]["additional_fields"].update({"uuid": new_uuid})
            parents.append(static_fw)
            all_fws.append(static_fw)
        # Analysis FW
        edl_fw = Firework(
            EDLAnalysis(
                uuids=uuids, replace_uuid=self.fw_spec["uuid"], db_file=">>db_file<<"
            ),
            name="EDL_analysis",
            parents=parents,
        )
        all_fws.append(edl_fw)
        wf = Workflow(all_fws, name="EDL_WF")
        return wf

    def run_task(self, fw_spec):

        # Variables
        is_bulk = self["is_bulk"]
        counter = self["counter"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        vasp_cmd = self["vasp_cmd"]

        # Connect to DB
        client = VaspCalcDb.from_db_file(db_file, admin=True)
        db = client.db

        # Get the launch_id for the parent_FW
        launch_id = self.launchpad.fireworks.find_one({"fw_id": self.fw_id})[
            "launches"
        ][0]

        # Parent Directory name
        parent_dir_name = self.launchpad.launches.find_one({"launch_id": launch_id})[
            "launch_dir"
        ]
        # Check whether the FW hits a wall_time
        try:
            wall_time_reached_errors = [
                correction["errors"][0] == "Walltime reached"
                for correction in db["tasks"].find_one({"uuid": fw_spec["uuid"]})[
                    "custodian"
                ][0]["corrections"]
            ]
        except (KeyError, TypeError) as e:
            print(f"{e}: Had trouble detecting errors in VaspRun...")
            wall_time_reached_errors = []
            pass
        # Imtermediate nodes that mutate the workflow
        if counter < fw_spec["max_tries"] and any(wall_time_reached_errors):

            if counter == 0:  # root node
                uuid_lineage = []
            else:
                uuid_lineage = fw_spec["uuid_lineage"]  # inherit from parent

            # Retrieving latest geometry from parent
            structure = Structure.from_dict(
                db["tasks"].find_one({"uuid": fw_spec["uuid"]})["output"]["structure"]
            )
            # Slab object of parent
            if not fw_spec["is_bulk"]:
                slab = Slab.from_dict(
                    db["tasks"].find_one({"uuid": fw_spec["uuid"]})["slab"]
                )

            # Retriving magnetic moments from parent
            magmoms = structure.site_properties["magmom"]

            # counts
            counter += 1
            # vasp_input_set of parent, to be inherited by child, except for magmoms and structure
            vasp_input_set_parent_dict = fw_spec["_tasks"][0]["vasp_input_set"]
            # Update structure and magmoms tied to the parent input set with that of the child
            vasp_input_set_parent_dict["structure"] = structure.as_dict()
            vasp_input_set_parent_dict["user_incar_settings"] = {"MAGMOM": magmoms}
            vasp_input_set_parent_updated_struct_magmoms = MOSurfaceSet.from_dict(
                vasp_input_set_parent_dict
            )

            # Create a unique uuid for child
            fw_new_uuid = uuid.uuid4()
            uuid_lineage.append(fw_spec["uuid"])  # UUID provenance for downstream nodes
            # OptimizeFW for child
            fw_new = OptimizeFW(
                name=fw_spec["name"],
                structure=structure,
                max_force_threshold=None,
                vasp_input_set=vasp_input_set_parent_updated_struct_magmoms,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
                job_type="normal",
                spec={
                    "counter": counter,
                    "_add_launchpad_and_fw_id": True,
                    "_pass_job_info": True,
                    "uuid_lineage": uuid_lineage,
                    "uuid": fw_new_uuid,
                    "max_tries": fw_spec["max_tries"],
                    "name": fw_spec["name"],  # pass parent name to child
                    "wall_time": fw_spec["wall_time"],
                    "is_bulk": True if fw_spec["is_bulk"] else False,
                    "is_adslab": fw_spec.get("is_adslab"),
                },
            )

            # Appending extra tasks
            fw_new.tasks[3]["additional_fields"].update({"uuid": fw_new_uuid})

            # Disable gunzip in RunVaspCustodian
            fw_new.tasks[1].update({"gzip_output": False})

            # Insert a CopyFilesFromCalcLoc Task into the childFW to inherit
            fw_new.tasks.insert(
                1, CopyFiles(from_dir=parent_dir_name, files_to_copy=["WAVECAR"])
            )

            # Insert a DeleteFiles Task into the childFW to delete previous WAVECAR
            fw_new.tasks.insert(
                2, DeleteFilesPrevFolder(files=["WAVECAR"], calc_dir=parent_dir_name)
            )

            # GzipPrevFolder
            # fw_new.tasks.insert(3, GzipPrevDir(calc_dir=parent_dir_name))

            fw_new.tasks.append(
                ContinueOptimizeFW(
                    is_bulk=is_bulk, counter=counter, db_file=db_file, vasp_cmd=vasp_cmd
                )
            )

            # Make sure that the child task doc from VaspToDB has the "Slab" object with wyckoff positions
            if counter > 0 and not fw_spec["is_bulk"]:
                fw_new.tasks[5]["additional_fields"].update({"slab": slab})

            # Get the environment that the parent ran on (either laikapack or nersc for now) and enforce that
            # child runs on the same resource/filesystem. Additionally, if the root ran on laikapack and
            # job triggered walltime handler, then the child can relinquish wall_time constraints
            import os

            if "nid" in os.environ["HOSTNAME"] or "cori" in os.environ["HOSTNAME"]:
                fw_new.tasks[3].update({"wall_time": fw_spec["wall_time"]})
                host = (
                    "nersc"  # this needs to be in the fworker config as query on nersc
                )
            elif "mo-wflow" in os.environ["HOSTNAME"]:
                # Switch off wall-time handling in child
                fw_new.spec["wall_time"] = None
                fw_new.tasks[3].update({"wall_time": None})
                host = "laikapack"  # should be in laikapack config

            # Pin the children down to the same filesystem as the root
            fw_new.spec["host"] = host

            fw_new.tasks[5].update(
                {"defuse_unsuccessful": False}
            )  # Allow continuation in child job

            # Bulk Continuation
            if is_bulk:
                return FWAction(detours=[fw_new])

            # Slab Continuation
            elif not is_bulk and not fw_spec.get("is_adslab"):
                fw_new.spec["oriented_uuid"] = fw_spec["oriented_uuid"]
                return FWAction(detours=[fw_new])

            # Adslab Continuation
            elif fw_spec["is_adslab"]:
                fw_new.spec["oriented_uuid"] = fw_spec["oriented_uuid"]
                fw_new.spec["slab_uuid"] = fw_spec["slab_uuid"]
                return FWAction(detours=[fw_new])

        # Terminal node
        else:
            if is_bulk:
                #               TODO: Possible by detours
                #               fw_spec["_tasks"].append(GzipDir().to_dict())
                #               self.launchpad.fireworks.find_one_and_update(
                #                   {"fw_id": self.fw_id}, {"$set": {"spec._tasks": fw_spec["_tasks"]}}
                #               )
                return FWAction(
                    update_spec={"oriented_uuid": fw_spec["uuid"]}, propagate=True
                )

            elif not is_bulk and not fw_spec.get("is_adslab"):
                self.task_doc = db["tasks"].find_one({"uuid": fw_spec["uuid"]})
                self.fw_spec = fw_spec
                wf = self.run_EDL()
                return FWAction(
                    detours=[wf],
                    update_spec={
                        "oriented_uuid": fw_spec["oriented_uuid"]
                        if "oriented_uuid" in fw_spec
                        else None,
                        "slab_uuid": fw_spec["uuid"],
                    },
                )
            elif fw_spec["is_adslab"]:
                self.task_doc = db["tasks"].find_one({"uuid": fw_spec["uuid"]})
                self.fw_spec = fw_spec
                wf = self.run_EDL()
                return FWAction(
                    detours=[wf],
                    update_spec={
                        fw_spec["name"]: {
                            "oriented_uuid": fw_spec["oriented_uuid"],
                            "slab_uuid": fw_spec["slab_uuid"],
                            "adslab_uuid": fw_spec["uuid"],
                        }
                    },
                )
