import json
import uuid

from pydash.objects import has, get

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)


# Help function to avovid tuples as key.
def json_format(dt):
    dt_new = {}
    for k, v in dt.items():
        k_str = "".join(map(str, k))
        dt_new.update({k_str: v})
    return dt_new


@explicit_serialize
class WulffShapeFW(FiretaskBase):
    """
    FireTask to do the Wulff-Shape Analysis.

    Args:
        tag: datetime string for folder name and id.
        db_file: database file path
        slab_formula: Reduced formula of the slab model e.g (RuO2)
        miller_index: Crystallographic orientations of the slab model.
        wulff_plot (default: False): Get automatically the wulffshape plot.
        to_db (default: True): Save the data on the db or in a json_file.

    return:
        summary_dict (JSON) with Wulff-Shape information inside.
    """

    required_params = ["bulk_structure", "db_file"]
    optional_params = ["wulff_plot", "to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
        wulff_plot = self.get("wulff_plot", True)
        bulk_structure = self["bulk_structure"]
        Ev2Joule = 16.0219  # eV/Angs2 to J/m2
        summary_dict = {}

        # new uuid for the wulff-shape
        wulff_shape_uuid = uuid.uuid4()
        summary_dict["wulff_uuid"] = wulff_shape_uuid

        # Bulk formula
        bulk_formula = bulk_structure.composition.reduced_formula

        # Connect to DB and Surface Energies collection
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        collection = mmdb.db["surface_energies"]

        # Find Surface energies for the given material + facet
        # docs = collection.find({"task_label": {"$regex": "^{}".format(bulk_formula)}})
        reduced_formula_miller_indices = [
            k for k in fw_spec if bulk_formula in k
        ]  # retrieve all the keys with reduced_formula from fw_spec
        docs = [
            collection.find_one(
                {
                    "oriented_uuid": fw_spec[rfmi]["oriented_uuid"],
                    "slab_uuid": fw_spec[rfmi]["slab_uuid"],
                }
            )
            for rfmi in reduced_formula_miller_indices
        ]

        # Surface energy and structures dictionary
        surface_energies_dict = {}
        structures_dict = {}
        for (
            d
        ) in (
            docs
        ):  # FIXME: We need to update the dict with the lowest energy facet if there are multiple terminations
            slab_struct = d["slab_struct"]  # as dict
            miller_index = tuple(map(int, d["miller_index"]))
            # Round to 4 decimals
            surface_energy = round(d["surface_energy"] * Ev2Joule, 4)
            if surface_energy < 0:
                continue  # skip negative surface_energies
            if (
                miller_index not in surface_energies_dict
                or surface_energy < surface_energies_dict[miller_index]
            ):
                surface_energies_dict.update({miller_index: surface_energy})
                structures_dict.update({d["miller_index"]: slab_struct})

        breakpoint()
        # Wulff Analysis
        wulffshape_obj, wulff_info, area_frac_dict = self.get_wulff_analysis(
            bulk_structure, surface_energies_dict
        )

        # Store data on summary_dict
        summary_dict["task_label"] = "{}_wulff_shape_{}".format(
            bulk_formula, wulff_shape_uuid
        )
        summary_dict["surface_energies"] = json_format(surface_energies_dict)
        summary_dict["wulff_info"] = wulff_info
        summary_dict["area_fractions"] = area_frac_dict
        summary_dict["slab_structures"] = structures_dict

        breakpoint()

        # Plot
        if wulff_plot:
            w_plot = wulffshape_obj.get_plot()
            w_plot.savefig("{}_wulff_shape.png".format(bulk_formula), dpi=100)

        # Add results to db
        if to_db:
            mmdb.collection = mmdb.db["{}_wulff_shape_analysis".format(bulk_formula)]
            mmdb.collection.insert_one(summary_dict)

        else:
            with open("{}_wulff_shape_analysis.json".format(bulk_formula), "w") as f:
                f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # Logger
        logger.info("Wulff-Shape Analysis, Done!")

        # Send the uuid as unique wulff-shape
        return FWAction(
            update_spec={"wulff_uuid": wulff_shape_uuid},
            propagate=True,
        )

    def get_wulff_analysis(self, bulk_structure, surface_energies_dict):
        """
        Makes the wulff analysis as a function of bulk lattice, facets
        and their surface energy.

        Args:
            bulk_structure (Structure): Easy way to get lattice paramenters.
            surface_enegies_dict (dict): {hkl: surface_energy (J/m2)}

        Return:
            Wulff Shape Analysis information.
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.analysis.wulff import WulffShape

        # Conventional standard structure
        SGA = SpacegroupAnalyzer(bulk_structure)
        bulk_struct = SGA.get_conventional_standard_structure()

        # Get key/values from energies dict J/m^2
        miller_list = surface_energies_dict.keys()
        e_surf_list = surface_energies_dict.values()

        # WulffShape Analysis
        wulffshape_obj = WulffShape(bulk_struct.lattice, miller_list, e_surf_list)

        # Collect wulffshape properties
        shape_factor = wulffshape_obj.shape_factor
        anisotropy = wulffshape_obj.anisotropy
        weight_surf_energy = wulffshape_obj.weighted_surface_energy  # J/m2
        shape_volume = wulffshape_obj.volume
        effective_radius = wulffshape_obj.effective_radius
        area_frac_dict = json_format(
            wulffshape_obj.area_fraction_dict
        )  # {hkl: area_hkl/total area on wulff}

        wulff_info = {
            "shape_factor": shape_factor,
            "anisotropy": anisotropy,
            "weight_surf_energy": weight_surf_energy,
            "volume": shape_volume,
            "effective_radius": effective_radius,
        }

        return wulffshape_obj, wulff_info, area_frac_dict
