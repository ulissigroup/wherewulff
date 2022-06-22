import numpy as np
from pydash.objects import has, get
import uuid

from pymatgen.core.structure import Structure

from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.fireworks.core import StaticFW

from WhereWulff.dft_settings.settings import MOSurfaceSet


@explicit_serialize
class StaticBulkFireTask(FiretaskBase):
    """
    Equilibrium Bulk structure StaticFW-

    Args:
        reduced_formula (e.g RuO2) : structure composition as reduced formula.
        bulks                      : Equilibrium bulks from EOS_fitting.
        vasp_cmd                   : Environment variable for VASP.
        db_file                    : To connect to the DB.


    Returns:
        Static (Single-point) calculation of each equilibrium structure.
    """

    required_params = ["reduced_formula", "bulks", "vasp_cmd", "db_file"]
    optional_params = []

    def run_task(self, fw_spec):

        # Variables
        reduced_formula = self["reduced_formula"]
        bulks = self["bulks"]
        vasp_cmd = self["vasp_cmd"]
        db_file = env_chk(self.get("db_file"), fw_spec)

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # StaticBulk
        if bulks is None:
            # Get equilibrium bulk from DB
            # Retrieve the uuids for the EOS FireTasks from the spec
            eos_uuids = [fw_spec[k] for k in fw_spec if "eos_uuid" in k]
            collection = mmdb.db[f"{reduced_formula}_eos"]
            bulk_metadata_docs = [
                collection.find_one({"task_label": eos_uuid}) for eos_uuid in eos_uuids
            ]

            bulk_candidates = {"magnetic_order": [], "structure": [], "energy": []}
            for d in bulk_metadata_docs:
                magnetic_ordering = d["magnetic_ordering"]
                structure_eq = Structure.from_dict(d["structure_eq"])
                energy_eq = d["energy_eq"]
                bulk_candidates["magnetic_order"].append(magnetic_ordering)
                bulk_candidates["structure"].append(structure_eq.as_dict())
                bulk_candidates["energy"].append(energy_eq)
            # Generate a set of StaticFW additions that will calc. DFT energy
            bulk_static_fws = []
            bulk_static_uuids = {}
            for magnetic_order, struct in zip(
                bulk_candidates["magnetic_order"], bulk_candidates["structure"]
            ):
                # Create unique uuid for each StaticFW
                static_bulk_uuid = uuid.uuid4()
                struct = Structure.from_dict(struct)
                vasp_input_set = MOSurfaceSet(
                    struct, user_incar_settings={"NSW": 0}, bulk=True
                )
                name = f"{struct.composition.reduced_formula}_{magnetic_order}_static_energy"
                bulk_static_fw = StaticFW(
                    struct,
                    name=name,
                    vasp_input_set=vasp_input_set,
                    vasp_cmd=vasp_cmd,
                    db_file=db_file,
                )
                bulk_static_fw.tasks[3]["additional_fields"].update(
                    {
                        "magnetic_ordering": magnetic_order,
                        "static_bulk_uuid": static_bulk_uuid,
                    }
                )
                # Pass the static_bulk_uuid to the bulk_stability FW
                bulk_static_uuids[
                    f"static_bulk_uuid_{magnetic_order}"
                ] = static_bulk_uuid
                #                bulk_static_fw.tasks[3].update(
                #                    {
                #                        "task_fields_to_push": {
                #                            f"static_bulk_uuid_{magnetic_order}": static_bulk_uuid
                #                        }
                #                    }
                #                )
                bulk_static_fws.append(bulk_static_fw)

        return FWAction(
            detours=bulk_static_fws,
            update_spec={"bulk_static_dict": bulk_static_uuids},
            propagate=True,
        )
