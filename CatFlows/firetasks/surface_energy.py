import json

from pydash.objects import has, get

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import Slab

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)


@explicit_serialize
class SurfaceEnergyFireTask(FiretaskBase):
    """
    Computes the surface energy for stoichiometric slab models.

    Args:
        slab_formula: Reduced formula of the slab model e.g (RuO2)
        miller_index: Crystallographic orientations of the slab model.
        db_file: database file path
        to_db (default: True): Save the data on the db or in a json_file.

    return:
        summary_dict (DB/JSON) with surface energy information.
    """

    required_params = ["slab_formula", "miller_index", "db_file"]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        slab_formula = self["slab_formula"]
        miller_index = self["miller_index"]
        oriented_uuid = fw_spec.get("oriented_uuid")
        slab_uuid = fw_spec.get("slab_uuid")
        to_db = self.get("to_db", True)
        summary_dict = {
            "task_label": "{}_{}_surface_energy".format(slab_formula, miller_index),
            "slab_formula": slab_formula,
            "miller_index": miller_index,
            "oriented_uuid": oriented_uuid,
            "slab_uuid": slab_uuid,
        }

        # Collect and store tasks_ids
        all_task_ids = []

        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        oriented = mmdb.collection.find_one({"uuid": oriented_uuid})
        slab = mmdb.collection.find_one({"uuid": slab_uuid})

        all_task_ids.append(oriented["uuid"])
        all_task_ids.append(slab["uuid"])

        # Get Structures from DB
        oriented_struct = Structure.from_dict(
            oriented["calcs_reversed"][-1]["output"]["structure"]
        )
        slab_struct = Structure.from_dict(
            slab["calcs_reversed"][-1]["output"]["structure"]
        )

        # Get DFT Energies from DB
        oriented_E = oriented["calcs_reversed"][-1]["output"]["energy"]
        slab_E = slab["calcs_reversed"][-1]["output"]["energy"]

        # Build Slab Object
        slab_obj = Slab(
            slab_struct.lattice,
            slab_struct.species,
            slab_struct.frac_coords,
            miller_index=list(map(int, miller_index)),
            oriented_unit_cell=oriented_struct,
            shift=0,
            scale_factor=0,
            energy=slab_E,
        )

        slab_Area = slab_obj.surface_area

        # Formulas
        oriented_formula = oriented_struct.composition.reduced_formula
        slab_formula = slab_struct.composition.reduced_formula

        # Compositions
        bulk_comp = oriented_struct.composition.as_dict()
        slab_comp = slab_struct.composition.as_dict()

        bulk_unit_form_dict = Composition(
            {el: bulk_comp[el] for el in bulk_comp.keys() if el != "O"}
        ).as_dict()
        slab_unit_form_dict = Composition(
            {el: slab_comp[el] for el in bulk_comp.keys() if el != "O"}
        ).as_dict()

        bulk_unit_form = sum(bulk_unit_form_dict.values())
        slab_unit_form = sum(slab_unit_form_dict.values())
        slab_bulk_ratio = slab_unit_form / bulk_unit_form

        # Calc. surface energy - Assumes symmetric
        surface_energy = self.get_surface_energy(slab_E, oriented_E, slab_bulk_ratio, slab_Area)

        # Surface energy for non-dipolar, symmetric and stoichiometric
#       if (
#           not slab_obj.is_polar()
#           and slab_obj.is_symmetric()
#           and slab_formula == oriented_formula
#       ):
#           surface_energy = self.get_surface_energy(
#               slab_E, oriented_E, slab_bulk_ratio, slab_Area
#           )

#       else:
#           surface_energy = None

        # Summary dict
        summary_dict["oriented_struct"] = oriented_struct.as_dict()
        summary_dict["slab_struct"] = slab_struct.as_dict()
        summary_dict["oriented_E"] = oriented_E
        summary_dict["slab_E"] = slab_E
        summary_dict["slab_Area"] = slab_Area
        summary_dict["is_polar"] = str(slab_obj.is_polar())
        summary_dict["is_symmetric"] = str(slab_obj.is_symmetric())
        if slab_formula == oriented_formula:
            summary_dict["is_stoichiometric"] = str(True)
        else:
            summary_dict["is_stoichiometric"] = str(False)

        summary_dict["N"] = slab_bulk_ratio
        summary_dict["surface_energy"] = surface_energy

        # Add results to db
        if to_db:
            mmdb.collection = mmdb.db["surface_energies"]
            mmdb.collection.insert_one(summary_dict)

        else:
            with open(
                "{}_{}_surface_energy.json".format(slab_formula, miller_index), "w"
            ) as f:
                f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # Logger
        logger.info(
            "{}_{} Surface Energy: {} [eV/A**2]".format(
                slab_formula, miller_index, surface_energy
            )
        )

        # Send the summary_dict to the child FW
        return FWAction(
            update_spec={
                f"{slab_formula}_{miller_index}": {
                    "oriented_uuid": oriented_uuid,
                    "slab_uuid": slab_uuid,
                }
            },
            propagate=True,
        )

    def get_surface_energy(self, slab_E, oriented_E, slab_bulk_ratio, slab_Area):
        """
        Surface energy for non-dipolar, symmetric and stoichiometric
        Units: eV/A**2

        Args:
            slab_E: DFT energy from slab optimization [eV]
            oriented_E: DFT energy from oriented bulk optimization [eV]
            slab_bulk_ratio: slab units formula per bulk units formula
            slab_area: Area from the slab model XY plane [A**2]
        Return:
            gamma_hkl - Surface energy for symmetric and stoichiometric model.
        """
        gamma_hkl = (slab_E - (slab_bulk_ratio * oriented_E)) / (
            2 * slab_Area
        )  # scaling for bulk!
        return gamma_hkl
