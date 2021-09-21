import uuid

from pydash.objects import has, get

from pymatgen.core import Structure

from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.database import VaspCalcDb
from atomate.common.firetasks.glue_tasks import CopyFiles, DeleteFilesPrevFolder

from CatFlows.dft_settings.settings import MOSurfaceSet


@explicit_serialize
class ContinueOptimizeFW(FiretaskBase):
    """
    Custom OptimizeFW Firetask that handles wall-time issues

    Args:

    Returns:

    """

    required_params = ["is_bulk", "counter", "vasp_cmd"]
    optional_params = ["db_file"]

    def run_task(self, fw_spec):

        # Variables
        is_bulk = self["is_bulk"]
        counter = self["counter"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        vasp_cmd = self["vasp_cmd"]

        # Connect to DB
        db = VaspCalcDb.from_db_file(db_file, admin=True)

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
            # Retrieving structure from parent
            structure = Structure.from_dict(
                db["tasks"].find_one({"uuid": fw_spec["uuid"]})["output"]["structure"]
            )
            # Retriving magnetic moments from parent
            magmoms = structure.site_properties["magmom"]
            # counts
            counter += 1
            vasp_input_set = MOSurfaceSet(
                structure, psp_version="PBE_54", bulk=True if is_bulk else False
            )
            # initial_magmoms=magmoms) # FIXME

            # Create a unique uuid for child
            fw_new_uuid = uuid.uuid4()

            # OptimizeFW for child
            fw_new = OptimizeFW(
                name=fw_spec["name"],
                structure=structure,
                max_force_threshold=None,
                vasp_input_set=vasp_input_set,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
                job_type="normal",
                spec={
                    "counter": counter,
                    "_add_launchpad_and_fw_id": True,
                    "_pass_job_info": True,
                    "uuid": fw_new_uuid,
                    "max_tries": fw_spec["max_tries"],
                    "name": fw_spec["name"],  # pass parent name to child
                    "wall_time": fw_spec["wall_time"],
                    "is_bulk": True if fw_spec["is_bulk"] else False,
                    "is_adslab": fw_spec.get("is_adslab"),
                },
            )

            # Appending extra tasks
            fw_new.tasks[1].update({"wall_time": fw_spec["wall_time"]})
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
            fw_new.tasks.append(ContinueOptimizeFW())

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

        # Terminal node
        else:
            if is_bulk:
                return FWAction(update_spec={"oriented_uuid": fw_spec["uuid"]})

            elif not is_bulk and not fw_spec.get("is_adslab"):
                return FWAction(
                    update_spec={
                        "oriented_uuid": fw_spec["oriented_uuid"],
                        "slab_uuid": fw_spec["uuid"],
                    }
                )
            elif fw_spec["is_adslab"]:
                return FWAction(
                    update_spec={
                        fw_spec["name"]: {
                            "oriented_uuid": fw_spec["oriented_uuid"],
                            "slab_uuid": fw_spec["slab_uuid"],
                            "adslab_uuid": fw_spec["uuid"],
                        }
                    }
                )
