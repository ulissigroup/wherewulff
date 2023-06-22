from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.vasp.database import VaspCalcDb
from atomate.utils.utils import env_chk
from pymatgen.core import Structure
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


@explicit_serialize
class EDLAnalysis(FiretaskBase):

    # required_params = ["uuids", "slab_uuid", "db_file", "nelects"]

    def run_task(self, fw_spec):
        """Placeholder firetask for fit Free energies versus applied potential for
        PZC and free energy at a specific reaction condition determination"""

        db_file = env_chk(">>db_file<<", fw_spec)
        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        task_collection = mmdb.db["tasks"]
        uuids = fw_spec["_tasks"][0]["uuids"]
        workfunctions = []
        nelects = []
        USHEs = []
        avg_pots = []
        escfs = []
        for uuid in uuids:
            task_doc = task_collection.find_one({"uuid": uuid})
            nelect = task_doc["calcs_reversed"][0]["output"]["outcar"]["nelect"]
            efermi = task_doc["calcs_reversed"][0]["output"]["efermi"]
            escf = task_doc["calcs_reversed"][0]["output"]["ionic_steps"][-1][
                "e_0_energy"
            ]
            escfs.append(escf)
            xy_averaged_locpot = task_doc["calcs_reversed"][0]["output"]["locpot"]["2"]
            xaxis = np.linspace(
                0,
                Structure.from_dict(
                    task_doc["calcs_reversed"][0]["output"]["structure"]
                ).lattice.matrix[2][2],
                len(xy_averaged_locpot),
            )
            avg_pot = np.average(
                np.array(xy_averaged_locpot)[(xaxis > 0) & (xaxis < 5)]
            )
            avg_pots.append(avg_pot)
            workfunction = avg_pot - efermi
            USHE = -4.6 + workfunction
            USHEs.append(USHE)
            workfunctions.append(workfunction)
            nelects.append(nelect)
        charges = np.array(nelects) - np.median(nelects)
        corrections = charges * np.array(USHEs) + charges * np.array(avg_pots)
        free_energies = np.array(escfs) + corrections
        # Fit the free energies and the USHEs
        def func(USHE, C, U0, E0):
            Ghat = -0.5 * C * (USHE - U0) ** 2 + E0
            return Ghat

        Us = np.arange(min(USHEs), max(USHEs), 0.1)
        fit_params = curve_fit(func, xdata=np.array(USHEs), ydata=free_energies)[0]
        print(fit_params)
        Ghats = func(Us, *fit_params)
        fig = plt.figure(figsize=(10.0, 10.0))
        ax = fig.add_subplot(111)
        ax.plot(Us, Ghats, "b-")
        ax.scatter(USHEs, free_energies, s=35, c="r")
        ax.set_xlabel("U (V SHE)")
        ax.set_ylabel("Free energy (eV)")
        ax.text(
            fit_params[1] - 0.5,
            fit_params[2] - 3,
            "["
            + str(round(fit_params[1], 2))
            + ","
            + str(round(fit_params[2], 2))
            + "]",
            color="green",
            fontsize=12,
        )
        fig.savefig("PZC_plot.png")

        breakpoint()

        return workfunction
