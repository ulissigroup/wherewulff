"""
Copyright (c) 2022 Carnegie Mellon University.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import uuid

from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.analysis.eos import EOS

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)


@explicit_serialize
class FitEquationOfStateFW(FiretaskBase):
    """
    Automated analysis task to fit an EOS after transmutter workflow.

    Args:
        eos     (default: vinet): Selecting the EOS used for fitting E vs Vol.
        plot    (default: True) : Exporting automatically EOS plot.
        to_db   (default: True) : Adding new data to EOS collection.
        db_file                 : Environment variable to connect to the DB.

    Return:
        Equilibrium structure for a given Bulk using
        (Energy vs Volume) deformations.

    """

    required_params = ["magnetic_ordering", "db_file"]
    optional_params = ["eos", "plot", "to_db"]

    def run_task(self, fw_spec):

        # Variables
        bulk_uuid = fw_spec.get("oriented_uuid")
        magnetic_ordering = self["magnetic_ordering"]
        eos = self.get("eos", "vinet")
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        plot = self.get("plot", True)
        summary_dict = {"eos": eos, "magnetic_ordering": magnetic_ordering}

        # new uuid for the eos-analysis
        eos_uuid = uuid.uuid4()
        summary_dict["eos_uuid"] = eos_uuid

        # Connect to DB
        all_task_ids = []
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Find optimization through bulk_uuid
        d = mmdb.collection.find_one({"uuid": bulk_uuid})

        # Get geometry from optimization
        structure_dict = d["calcs_reversed"][-1]["output"]["structure"]
        structure = Structure.from_dict(structure_dict)
        pretty_formula = structure.composition.reduced_formula
        all_task_ids.append(d["task_id"])
        # Retriving bulk deformations
        # Get all the UUIDs that were sent from the deformation FWs
        deformation_uuids = [fw_spec[k] for k in fw_spec if "deformation" in k]
        docs = [
            mmdb.collection.find_one({"deform_uuid": deform_uuid})
            for deform_uuid in deformation_uuids
        ]
        # Store optimized Bulk and pretty-formula
        summary_dict["structure_orig"] = structure.as_dict()
        summary_dict["formula_pretty"] = pretty_formula

        # Get (energy, volume) from the deformations
        energies, volumes = [], []
        for i, d in enumerate(docs):
            if i == 0:
                magmoms_list = d["orig_inputs"]["incar"][
                    "MAGMOM"
                ]  # Store the magmoms from the first deformation
            # Assert that the deformations are all consistent in their orderings
            assert magmoms_list == d["orig_inputs"]["incar"]["MAGMOM"]
            s = Structure.from_dict(d["calcs_reversed"][-1]["output"]["structure"])
            energies.append(d["calcs_reversed"][-1]["output"]["energy"])
            volumes.append(s.volume)
            all_task_ids.append(d["task_id"])

        # Append to summary_dict
        summary_dict["energies"] = energies
        summary_dict["volumes"] = volumes
        summary_dict["all_task_ids"] = all_task_ids

        # Fit the equation-of-states
        eos = EOS(eos_name=eos)
        eos_fit = eos.fit(volumes, energies)
        summary_dict["volume_eq"] = eos_fit.v0
        summary_dict["energy_eq"] = eos_fit.e0

        # Scale optimized structure to the equilibrium volume
        structure.scale_lattice(eos_fit.v0)
        # Decorate the optimized structure with the relevant magmoms as a property
        structure.add_site_property("magmom", magmoms_list)
        summary_dict["structure_eq"] = structure.as_dict()

        # Store data on summary_dict
        summary_dict[
            "task_label"
        ] = f"{pretty_formula}_{magnetic_ordering}_eos_{eos_uuid}"

        # Add results to db or json file
        if to_db:
            mmdb.collection = mmdb.db[f"{pretty_formula}_eos"]
            mmdb.collection.insert_one(summary_dict)
        
        # Export json file
        with open(f"{pretty_formula}_{magnetic_ordering}_eos_analysis.json", "w") as f:
            f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # Export plot
        if plot:
            eos_plot = eos_fit.plot()
            eos_plot.savefig(
                f"{pretty_formula}_{magnetic_ordering}_eos_plot.png", dpi=300
            )

        # logger
        logger.info("EOS Fitting Completed!")
        # Exit function by sending the eos_uuid to the static_bulk FireTask
        return FWAction(
            update_spec={f"eos_uuid_{magnetic_ordering}": summary_dict["task_label"]}
        )
