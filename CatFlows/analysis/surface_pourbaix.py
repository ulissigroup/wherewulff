import json
import uuid

import numpy as np

from pydash.objects import has, get

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)

@explicit_serialize
class SurfacePourbaixDiagramAnalyzer(FiretaskBase):
    """
    Post-processing FireTask to plot the surface pourbaix diagram
    out of OH/Ox full coverage terminations

    Args:
        bulk_structure: pymatgen structure for getting bulk-formula
        db_file       : defult db.json file
        to_db         : option whether send the data to DB or not.

    Returns:
        plot for each surface and DB json data.
    """

    requiered_params = ["bulk_structure", "db_file"]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        bulk_structure = self["bulk_structure"]
        summary_dict = {}

        # PBX Variables
        self.pH_range = list(range(0, 15, 1))
        kB = 0.0000861733 # eV/K
        Temp = 298.15 # Kelvin
        self.K = kB * Temp * np.log(10) # Nernst slope
        self.reference_energies = {"H2O":-14.25994015, "H2":-6.77818501}

        # Surface PBX diagram uuid
        surface_pbx_uuid = uuid.uuid4()
        summary_dict["surface_pbx_uuid"] = surface_pbx_uuid

        # Bulk formula
        bulk_formula = bulk_structure.composition.reduced_formula

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Find clean surfaces 
        collection_clean = mmdb.db["surface_energies"]
        docs_clean = collection_clean.find({"task_label": {"$regex": f"{bulk_formula}_.*_surface_energy"}})

        # Find OH/Ox terminations per surface
        collection_term = mmdb.db["tasks"]
        docs_OH = collection_term.find({"task_label": {"$regex": f"{bulk_formula}-.*-OH_.*"}})
        docs_Ox = collection_term.find({"task_label": {"$regex": f"{bulk_formula}-.*-O_.*"}})

        # Get miller indices and DFT Energies
        surface_clean_dict = {}
        for d_clean in docs_clean:
            miller_index = d_clean["miller_index"]
            dft_energy_clean = d_clean["slab_E"]
            surface_clean_dict.update({str(miller_index): dft_energy_clean})

        # List of (hkl)
        miller_indices = list(surface_clean_dict.keys())

        # Organizing OH termination by DFT energy - nested dict as {"hkl": {task_label: energy}}
        task_label_oh_dict = {}
        for hkl in miller_indices:
            task_label_oh_dict[hkl] = {}
            for d_OH in docs_OH:
                task_label = d_OH["task_label"]
                surf_orientation = task_label.split("-")[1]
                dft_energy_oh = d_OH["calcs_reversed"][-1]["output"]["energy"]
                task_label_oh_dict[surf_orientation].update({str(task_label): dft_energy_oh})

        # Find the lowest energy OH configuration per surface orientation - {task_label: energy}
        stable_oh_terminations = {}
        for hkl in miller_indices:
            oh_terminations = task_label_oh_dict[hkl]
            find_minimum = min(oh_terminations, key=oh_terminations.get)
            stable_oh_terminations.update({str(find_minimum): oh_terminations[find_minimum]})

        # Ox terminations
        stable_ox_termination = {}
        for hkl in miller_indices:
            for d_Ox in docs_Ox:
                task_label = d_Ox["task_label"]
                dft_energy_ox = d_Ox["calcs_reversed"][-1]["output"]["energy"]
                stable_ox_termination.update({str(task_label): dft_energy_ox})


    def oer_potential_std(self):
        """
        Standard OER bound --> H2O -> O2 + 4H+ + 4e-
        """
        return [(1.229 + self.K*pH)for pH in self.pH_range]

    def oer_potential_up(self, u_0=1.60):
        """
        OER bound, U_0 selected by user and/or Experimental conditions.
        
        default: 1.60 V
        """
        return [(u_0 + self.K*pH)for pH in self.pH_range]

    def _get_surface_potential_line(self, dft_energy_specie, dft_energy_reference, nH=4, nH2O=4):
        """
        Get line equation from:
            - clean --> OH:
                specie = OH-terminated
                reference = clean
            - OH --> Ox:
                specie = Ox-terminated
                reference = OH-terminated
            - nH: Released (H+ + e-) PCETs
            - nH2O: How many water molecules are adsorbed on top of the clean (e.g. 110 = 4)
        """
        intersept = (dft_energy_specie - dft_energy_reference - (nH2O*self.ref_energies["H2O"])) * (1/nH)
        intersept = intersept + (0.5 * self.ref_energies["H2"])
        return [(intersept + self.K*pH) for pH in self.pH_range]

    def _get_surface_pbx_diagram(self):
        """
        Matplotlib based function to build the PBX graph
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8,5))

        # Axis labels
        ax.set_xlabel("pH", fontsize=12, fontweight="bold")
        ax.set_ylabel("U$_{SHE}$ (V)", fontsize=12, fontweight="bold")

        # Axis Limits
        ax.set_title(f"{bulk_formula} ({miller_index})")
        ax.set_xlim(0.0, 14.0)
        ax.set_ylim(0.0, 2.50)

        # OER bounds for standard and selected OER conditions
        ax.plot(self.pH_range, oer_bound_std, linestyle="--", color="black")
        ax.plot(self.pH_range, oer_bound_up, linestyle="--", color="red")

        # Surface PBX boundaries
        ax.plot(self.pH_range, clean_2_OH, linestyle="-", color="blue")
        ax.plot(self.pH_range, OH_2_Ox, linstyle="-", color="red")

        # Fill surface-terminations regions
        ax.fill_between(self.pH_range, clean_2_OH, color="blue", alpha=0.2, label="clean")
        ax.fill_between(self.pH_range, clean_2_OH, OH_2_Ox, color="green", alpha=0.2, label="*OH")
        ax.fill_between(self.pH_range, OH_2_Ox, 2.50, color="red", alpha=0.2, label="*Ox")

        # Add legend
        plt.legend()

        return fig



        








        












        


