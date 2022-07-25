from fireworks import FiretaskBase, FWAction, explicit_serialize
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import env_chk
import numpy as np
import pandas as pd


@explicit_serialize
class analyzeUEffect(FiretaskBase):
    """We retrieve the relaxed slab at the +U and use that as a
    starting point for the adslab relaxations, with rotational sweep at
    that same value of U"""

    required_params = []

    def run_task(self, fw_spec):
        keys = [x for x in fw_spec if "-" in x and "_" in x]
        slab_adslab_uuids = [fw_spec[k] for k in keys]
        db_file = env_chk(DB_FILE, fw_spec)
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        # Go through all the keys and slab_adslab_uuids retrieving the energies and inputs and seeing if
        # things have converged
        results_dict = {}
        for k, s_ads_uuid in zip(keys, slab_adslab_uuids):
            #        breakpoint()
            print(k)
            slab_uuid = s_ads_uuid["slab_uuid"]
            adslab_uuid = s_ads_uuid["adslab_uuid"]
            has_slab_converged = mmdb.db["tasks"].find_one({"uuid": slab_uuid})[
                "calcs_reversed"
            ][0]["has_vasp_completed"]
            print(has_slab_converged, "slab_convergence")
            has_adslab_converged = mmdb.db["tasks"].find_one({"uuid": adslab_uuid})[
                "calcs_reversed"
            ][0]["has_vasp_completed"]
            print(has_adslab_converged, "adslab_convergence")
            # FIXME: Identify where the sensitivity analysis is being carried out by the non-zero value?
            slab_U_value = mmdb.db["tasks"].find_one({"uuid": slab_uuid})[
                "calcs_reversed"
            ][0]["input"]["incar"]["LDAUU"][0]
            adslab_U_value = mmdb.db["tasks"].find_one({"uuid": adslab_uuid})[
                "calcs_reversed"
            ][0]["input"]["incar"]["LDAUU"][0]
            if slab_U_value == adslab_U_value:
                U_value = str(slab_U_value)
            else:
                raise ValueError("Inconsistent U values!")
            print(U_value, "U")
            # Place the results in a dict
            slab_energy = mmdb.db["tasks"].find_one({"uuid": slab_uuid})[
                "calcs_reversed"
            ][0]["output"]["energy"]
            adslab_energy = mmdb.db["tasks"].find_one({"uuid": adslab_uuid})[
                "calcs_reversed"
            ][0]["output"]["energy"]
            if U_value in results_dict:
                if adslab_energy < results_dict[U_value]["adslab_energy"]:
                    results_dict[U_value]["adslab_energy"] = adslab_energy
                    results_dict[U_value]["has_adslab_converged"] = has_adslab_converged
                    results_dict[U_value]["adslab_uuid"] = adslab_uuid
            else:
                results_dict[U_value] = {
                    "slab_energy": slab_energy,
                    "adslab_energy": adslab_energy,
                    "has_slab_converged": has_slab_converged,
                    "slab_uuid": slab_uuid,
                    "adslab_uuid": adslab_uuid,
                    "has_adslab_converged": has_adslab_converged,
                }
        # Create pandas dict
        df = pd.DataFrame.from_dict(results_dict, orient="index")
        df_sort = df.sort_index()
        df_sort.to_csv("sensitivity_U_ads_energy.csv")

        # Access the tasks collection
        #        print(f"{fw_spec}")
        return FWAction(stored_data=results_dict)
