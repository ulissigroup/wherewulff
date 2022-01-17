import json
import uuid
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.pourbaix_diagram import (
    ELEMENTS_HO,
    PourbaixDiagram,
    PourbaixPlotter,
)

from pymatgen.analysis.structure_analyzer import OxideType
from pymatgen.ext.matproj import MPRester

from pymatgen.entries.compatibility import (
    MaterialsProjectAqueousCompatibility,
    MaterialsProjectCompatibility,
)

from pydash.objects import has, get

from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)


# Correction Dicts from MP2020
composition_correction = {
    "V": -1.7,
    "Cr": -1.999,
    "Mn": -1.668,
    "Fe": -2.256,
    "Co": -1.638,
    "Ni": -2.541,
    "W": -4.438,
    "Mo": -3.202,
}
oxide_type_correction = {"oxide": -0.687, "peroxide": -0.465, "superoxide": -0.161}

compat = MaterialsProjectCompatibility()


@explicit_serialize
class BulkStabilityAnalysis(FiretaskBase):
    """
    Automated Stability Analysis Task to directly get,
    Thermodynamic and electrochemical stability of a given material.

    Args:
        bulk_formula (e.g RuO2)   : structure composition as reduced formula
        db_file                   : To connect to the DB
        pbx_plot (default: True)  : Save .png in launcher folder for PbxDiagram
        ehull_plot                : Save .png in launcher folder for PhaseDiagram


    Returns:
        Stability Analysis to DB
    """

    required_params = ["reduced_formula", "db_file"]
    optional_params = ["pbx_plot"]

    def run_task(self, fw_spec):

        # Variables
        bulk_formula = self["reduced_formula"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        pbx_plot = self.get("pbx_plot", True)
        to_db = self.get("to_db", True)
        bulk_stability_uuid = str(uuid.uuid4())
        summary_dict = {"uuid": bulk_stability_uuid}

        # Connect to DB
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        # Get the static_bulk uuids from the static energy FWs
        bulk_static_uuids = [
            fw_spec["bulk_static_dict"][k] for k in fw_spec["bulk_static_dict"]
        ]

        # Retrieve from DB
        docs = [
            mmdb.collection.find_one({"static_bulk_uuid": sb_uuid})
            for sb_uuid in bulk_static_uuids
        ]
        # Get the doc with the lowest dft_energy
        d = sorted(docs, key=lambda x: x['calcs_reversed'][-1]['output']['energy'])[0] 

        # Collect data
        mag_label = d['magnetic_ordering']
        logger.info(f"Selecting {mag_label} as the most stable!")
        structure_dict = d["calcs_reversed"][0]["output"]["structure"]
        dft_energy = d["calcs_reversed"][0]["output"]["energy"]
        structure = Structure.from_dict(structure_dict)
        oxide_type = OxideType(structure).parse_oxide()[0]
        # Add the initial magmoms to the summary dict
        summary_dict["orig_magmoms"] = d["orig_inputs"]["incar"]["MAGMOM"]
        summary_dict["structure"] = structure.as_dict()
        summary_dict["formula_pretty"] = structure.composition.reduced_formula
        summary_dict["oxide_type"] = oxide_type
        summary_dict["uncorrected_energy"] = dft_energy
        # Get Correction
        bulk_composition = structure.composition
        comp_dict = {str(key): value for key, value in bulk_composition.items()}
        comp_dict_pbx = {
            str(key): value
            for key, value in bulk_composition.items()
            if key not in ELEMENTS_HO
        }
        correction = 0
        for i in list(comp_dict.keys()):
            if i in composition_correction:
                correction += composition_correction[i] * comp_dict[i]

        if oxide_type in list(oxide_type_correction.keys()):
            correction += oxide_type_correction[oxide_type] * comp_dict["O"]

        corrected_energy = dft_energy + correction
        summary_dict["correction"] = correction
        summary_dict["corrected_energy"] = corrected_energy

        # Parameters + data
        parameters = {}
        parameters["oxide_type"] = str(oxide_type)

        data = {}
        data["oxide_type"] = str(oxide_type)

        # Computed Entry
        computed_entry = ComputedEntry(
            composition=bulk_composition,
            energy=dft_energy,
            correction=correction,
            parameters=parameters,
            data=data,
            entry_id=f"{bulk_formula}_{mag_label}_{bulk_stability_uuid}",
        )

        # PhaseDiagram Analysis
        with MPRester() as mpr:
            # Get PhaseDiagram Entry
            phd_entry = PDEntry(
                bulk_composition, energy=corrected_energy, name="phd_entry"
            )
            # chemsys from comp
            chemsys = list(comp_dict.keys())

            # Check compatibility
            unprocessed_entries = mpr.get_entries_in_chemsys(chemsys)
            processed_entries = compat.process_entries(unprocessed_entries)
            processed_entries.append(phd_entry)

            # Build PhaseDiagram
            phd = PhaseDiagram(processed_entries)

            # Get PhD info
            eform_phd = phd.get_form_energy(phd_entry)
            eform_atom_phd = phd.get_form_energy_per_atom(phd_entry)
            e_hull_phd = phd.get_e_above_hull(phd_entry)

        # Store PhD data
        summary_dict["eform_phd"] = eform_phd
        summary_dict["eform_atom_phd"] = eform_atom_phd
        summary_dict["e_hull_phd"] = e_hull_phd

        # Pourbaix Diagram Analysis
        with MPRester() as mpr:
            # chemsys from compostion for pbx (no OH)
            chemsys = list(comp_dict_pbx.keys())

            # Get pbx_entries, pbx_entry and Ef/atom
            pbx_entries, pbx_entry, eform_atom_pbx = self.get_pourbaix_entries(
                mpr, computed_entry
            )

            pbx_entries.append(pbx_entry)

            # PBX Diagram
            pbx = PourbaixDiagram(
                pbx_entries, comp_dict=comp_dict_pbx, filter_solids=False
            )

            # Get electrochemical stability at conditions
            oer_stability = pbx.get_decomposition_energy(pbx_entry, pH=0, V=1.23)  # OER

        # Store PBX data
        summary_dict["eform_atom_pbx"] = eform_atom_pbx
        summary_dict["oer_stability"] = oer_stability

        # Get pbx plot
        if pbx_plot:
            plt = PourbaixPlotter(pbx).plot_entry_stability(
                pbx_entry, label_domains=True
            )
            plt.savefig(
                f"{bulk_formula}_{mag_label}_pbx.png", dpi=300
            )  # FIXME: Think this should be mag_label

        # Export json file
        with open(f"{bulk_formula}_{mag_label}_stability_analys.json", "w") as f:
            f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # To_DB
        if to_db:
            mmdb.collection = mmdb.db[f"{bulk_formula}_stability_analysis"]
            mmdb.collection.insert_one(summary_dict)

        # Logger
        logger.info("Stability Analysis Completed!")

    def get_pourbaix_entries(
        self, mpr, comp_entry, solid_compat="MaterialsProject2020Compatibility"
    ):
        """
        A helper function to get all entries necessary to generate PBX
        """
        import warnings
        import itertools
        from pymatgen.core.composition import Composition
        from pymatgen.core.periodic_table import Element
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.analysis.pourbaix_diagram import IonEntry, PourbaixEntry
        from pymatgen.core.ion import Ion
        from pymatgen.entries.compatibility import (
            Compatibility,
            MaterialsProject2020Compatibility,
            MaterialsProjectCompatibility,
        )

        # Selecting compatibility
        if solid_compat == "MaterialsProjectCompatibility":
            solid_compat = MaterialsProjectCompatibility()
        elif solid_compat == "MaterialsProject2020Compatibility":
            solid_compat = MaterialsProject2020Compatibility()
        elif isinstance(solid_compat, Compatibility):
            solid_compat = solid_compat

        # Comp_entry
        entry_composition = comp_entry.composition
        comp_dict = {
            str(key): value
            for key, value in entry_composition.items()
            if key not in ELEMENTS_HO
        }
        chemsys = list(comp_dict.keys())

        # Store PBX entries
        pbx_entries = []

        if isinstance(chemsys, str):
            chemsys = chemsys.split("-")

        # Get ion entries first, because certain ions have reference
        url = "/pourbaix_diagram/reference_data/" + "-".join(chemsys)
        ion_data = mpr._make_request(url)
        ion_ref_comps = [Composition(d["Reference Solid"]) for d in ion_data]
        ion_ref_elts = list(
            itertools.chain.from_iterable(i.elements for i in ion_ref_comps)
        )
        ion_ref_entries = mpr.get_entries_in_chemsys(
            list(set([str(e) for e in ion_ref_elts] + ["O", "H"])),
            property_data=["e_above_hull"],
            compatible_only=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You did not provide the required O2 and H2O energies.",
            )
            compat = MaterialsProjectAqueousCompatibility(solid_compat=solid_compat)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Failed to guess oxidation states.*")
            ion_ref_entries = compat.process_entries(ion_ref_entries)
        ion_ref_pd = PhaseDiagram(ion_ref_entries)

        # position the ion energies relative to most stable reference state
        for n, i_d in enumerate(ion_data):
            ion = Ion.from_formula(i_d["Name"])
            refs = [
                e
                for e in ion_ref_entries
                if e.composition.reduced_formula == i_d["Reference Solid"]
            ]
            if not refs:
                raise ValueError("Reference soldi not contained in entry list")
            stable_ref = sorted(refs, key=lambda x: x.data["e_above_hull"])[0]
            rf = stable_ref.composition.get_reduced_composition_and_factor()[1]

            solid_diff = (
                ion_ref_pd.get_form_energy(stable_ref)
                - i_d["Reference solid energy"] * rf
            )
            elt = i_d["Major_Elements"][0]
            correction_factor = ion.composition[elt] / stable_ref.composition[elt]
            energy = i_d["Energy"] + solid_diff * correction_factor
            ion_entry = IonEntry(ion, energy)
            pbx_entries.append(PourbaixEntry(ion_entry, "ion-{}".format(n)))

        # Construct the solid pourbaix entries from filtered ion_ref entries
        extra_elts = (
            set(ion_ref_elts)
            - {Element(s) for s in chemsys}
            - {Element("H"), Element("O")}
        )
        for entry in ion_ref_entries:
            entry_elts = set(entry.composition.elements)
            # Ensure no OH chemsys or extraneous elements from ion references
            if not (
                entry_elts <= {Element("H"), Element("O")}
                or extra_elts.intersection(entry_elts)
            ):
                # Create new computed entry
                form_e = ion_ref_pd.get_form_energy(entry)
                new_entry = ComputedEntry(
                    entry.composition, form_e, entry_id=entry.entry_id
                )
                pbx_entry = PourbaixEntry(new_entry)
                pbx_entries.append(pbx_entry)

        # New Computed Entry
        formation_energy = ion_ref_pd.get_form_energy(comp_entry)
        formation_energy_per_atom = ion_ref_pd.get_form_energy_per_atom(comp_entry)
        new_entry = ComputedEntry(
            comp_entry.composition, formation_energy, entry_id=comp_entry.entry_id
        )
        pbx_entry = PourbaixEntry(new_entry)

        return pbx_entries, pbx_entry, formation_energy_per_atom
