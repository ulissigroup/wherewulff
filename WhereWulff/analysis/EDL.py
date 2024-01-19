from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.vasp.database import VaspCalcDb
from atomate.utils.utils import env_chk
from pymatgen.core import Structure
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


@explicit_serialize
class EDLAnalysis(FiretaskBase):

    required_params = ["uuids", "replace_uuid", "db_file"]

    def run_task(self, fw_spec):
        """Placeholder firetask for fit Free energies versus applied potential for
        PZC and free energy at a specific reaction condition determination"""

        db_file = env_chk(self["db_file"], fw_spec)
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        task_collection = mmdb.db["tasks"]
        uuids = self["uuids"]
        replace_uuid = self["replace_uuid"]
        workfunctions = []
        nelects = []
        USHEs = []
        avg_pots = []
        escfs = []
        fermi_levels = []
        for uuid in uuids:
            breakpoint()
            # Need to find the latest task_id based on uuid and fw_id
            try:
                launch_id = mmdb.db["fireworks"].find_one(
                    {"spec._tasks.3.additional_fields.uuid": uuid}
                )["launches"][0]
            except IndexError:
                launch_id = mmdb.db["fireworks"].find_one(
                    {"spec._tasks.3.additional_fields.uuid": uuid}
                )["archived_launches"][-1]
            task_id = mmdb.db["launches"].find_one({"launch_id": launch_id})["action"][
                "stored_data"
            ]["task_id"]
            task_doc = task_collection.find_one({"task_id": task_id})
            nelect = task_doc["calcs_reversed"][0]["output"]["outcar"]["nelect"]
            breakpoint()
            nelects.append(nelect)
            efermi = task_doc["calcs_reversed"][0]["output"]["efermi"]
            fermi_levels.append(efermi)
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
        # FIXME: Need to add the fermi-shift for the charged slabs to reference to vacuum level in uncharged slab
        ref_fermi = np.median(fermi_levels)
        fermi_shifts = []
        breakpoint()
        for i, uuid in enumerate(uuids):
            fermi_shift = -avg_pots[i]
            fermi_shifts.append(fermi_shift)
            shifted_fermi = fermi_levels[i] + fermi_shift
            workfunction = 0 - shifted_fermi
            USHE = -4.6 + workfunction
            USHEs.append(USHE)
            workfunctions.append(workfunction)
        charges = np.array(nelects) - np.median(nelects)
        corrections = (
            # charges * np.array(USHEs) +
            # charges * np.array(avg_pots)
            # + np.array(fermi_shifts) * charges
            (-np.array(avg_pots) * charges)
            + np.array(workfunctions) * charges
        )
        free_energies = np.array(escfs) + corrections
        # Fit the free energies and the USHEs
        def func(USHE, C, U0, E0):
            Ghat = -0.5 * C * (USHE - U0) ** 2 + E0
            return Ghat

        breakpoint()
        Us = np.arange(min(USHEs), max(USHEs), 0.01)
        fit_params = curve_fit(
            func,
            xdata=np.array(USHEs),
            ydata=free_energies,
            maxfev=10000,
            # p0=[0.5, 0.2, -500],
        )[0]
        breakpoint()
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
        # We need to find the free energy at the reaction conditions and then mutate
        # the free energy in the task doc of that interface
        # ionic_steps = mmdb.db["tasks"].find_one({"uuid": replace_uuid})[
        #    "calcs_reversed"
        # ][0]["output"]["ionic_steps"]
        # initial_energy = ionic_steps[-1]["e_0_energy"]
        # final_energy = func([0.17], *fit_params)
        # print(
        #    f"Comparing doc with uuid: {replace_uuid} from {initial_energy} eV to {final_energy} eV"
        # )
        fig.savefig("PZC_plot.png")
        return FWAction(
            stored_data={
                #        "initial_energy": initial_energy,
                #        "energy_at_RC": final_energy,
                "USHEs": USHEs,
                "Ghats": free_energies,
                "wfs": workfunctions,
                "corrections": corrections,
                "escfs": escfs,
                "nelects": nelects,
                "fit_params": fit_params,
                "charges": charges,
                "avg_pots": avg_pots,
            }
        )
