import numpy as np
from pydash.objects import has, get

from pymatgen.core.structure import Structure

from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.utils.utils import env_chk
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.fireworks.core import StaticFW

from CatFlows.dft_settings.settings import MOSurfaceSet


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
            collection = mmdb.db[f"{reduced_formula}_eos"]
            bulk_metadata_docs = collection.find(
                {"task_label": {"$regex": f"{reduced_formula}_*_eos_*"}}
            )

            bulk_candidates = {"magnetic_order": [], "structure": [], "energy": []}
            for d in bulk_metadata_docs:
                magnetic_ordering = d["magnetic_ordering"]
                structure_eq = Structure.from_dict(d["structure_eq"])
                energy_eq = d["energy_eq"]
                bulk_candidates["magnetic_ordering"].append(magnetic_ordering)
                bulk_candidates["structure"].append(structure_eq.as_dict())
                bulk_candidates["energy"].append(energy_eq)

            # Generate a set of StaticFW additions that will calc. DFT energy
            bulk_static_fws = []
            for magnetic_order, struct in zip(
                bulk_candidates["magnetic_ordering"], bulk_candidates["structure"]
            ):
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
                    {"magnetic_ordering": magnetic_order}
                )
                bulk_static_fws.append(bulk_static_fw)

        return FWAction(detours=bulk_static_fws)
