import os
import json
from datetime import datetime
import numpy as np

from pydash.objects import has, get

import pymatgen
from pymatgen import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, get_symmetrically_distinct_miller_indices
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.io.vasp.inputs import Kpoints


from fireworks import FiretaskBase, FWAction, Workflow, explicit_serialize
from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk, get_meta_from_structure
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb


logger = get_logger(__name__)

# Setting magmom automatically
def set_magmom(structure):
    comp_dict = Composition(structure.formula).as_dict()
    mag_dict = Composition(structure.formula).as_dict()
    mag_dict['O'] = 0.6
    if 'H' in mag_dict.keys():
        mag_dict['H'] = 0.1
    for n in mag_dict.keys():
        if n != 'O' and n != 'H':
            mag_dict[n] = 5.0

    mag = []
    for k in comp_dict:
        n_sites = int(comp_dict[str(k)])
        magmom = mag_dict[str(k)]
        mag.append(str(n_sites)+'*'+str(magmom))

    mag_command = " ".join(mag)

    return mag_command

# Help function to avovid tuples as key.
def json_format(dt):
    dt_new = {}
    for k, v in dt.items():
        k_str = "".join(map(str, k))
        dt_new.update({k_str: v})
    return dt_new

# Theoretical Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """
    def __init__(self, structure, bulk=False, **kwargs):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, **kwargs)

        # Bulks are better with LREAL=.FALSE. ??
        if bulk:
            self._config_dict['INCAR'].update({'LREAL': False})
        else:
            self._config_dict['INCAR'].update({'LREAL': True})

    @property
    def incar(self):
        incar = super(MOSurfaceSet, self).incar

        #Setting Magnetic Moments  
        #magmom = set_magmom(self.structure)

        # Incar Settings for optimization
        incar_config = {"GGA": "RP", "ENCUT": 500, "EDIFF": 1e-5, "EDIFFG": -0.05, 
                        "ISYM": 0, "ISPIN": 2, "ISIF": 0}
        #update incar
        incar.update(incar_config)
        incar.update(self.user_incar_settings)

        return incar

    @property
    def kpoints(self):
        """
        Monkhorst-pack Gamma Centered scheme:
            bulks [50/a x 50/b x 50/c]
            slabs [30/a x 30/b x 1]
        """
        abc = np.array(self.structure.lattice.abc)

        if self.bulk:
            kpts = tuple(np.ceil(50.0 / abc).astype('int'))
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0,0,0))

        else:
            kpts = np.ceil(30.0 / abc).astype('int')
            kpts[2] = 1
            kpts = tuple(kpts)
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0,0,0))


@explicit_serialize
class SurfaceEnergyFW(FiretaskBase):
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
    required_params = ['slab_formula', 'miller_index', 'db_file']
    optional_params = ['to_db']

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        slab_formula = self["slab_formula"]
        miller_index = self["miller_index"]
        to_db = self.get("to_db", True)
        summary_dict = {"task_label": "{}_{}_surface_energy".format(slab_formula, miller_index),
                        "slab_formula": slab_formula, "miller_index": miller_index}

        # Collect and store tasks_ids
        all_task_ids = []

        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        oriented = mmdb.collection.find_one({'task_label': '{}_{} bulk optimization'.format(slab_formula, miller_index)})
        slab = mmdb.collection.find_one({'task_label': '{}_{} slab optimization'.format(slab_formula, miller_index)})

        all_task_ids.append(oriented['task_id'])
        all_task_ids.append(slab['task_id'])

        # Get Structures from DB
        oriented_struct = Structure.from_dict(oriented["calcs_reversed"][-1]["output"]['structure'])
        slab_struct = Structure.from_dict(slab["calcs_reversed"][-1]["output"]['structure'])

        # Get DFT Energies from DB
        oriented_E = oriented["calcs_reversed"][-1]["output"]["energy"]
        slab_E = slab["calcs_reversed"][-1]["output"]["energy"]

        # Build Slab Object
        slab_obj = Slab(slab_struct.lattice, slab_struct.species, slab_struct.frac_coords,
                        miller_index=list(map(int, miller_index)), oriented_unit_cell=oriented_struct, 
                        shift=0, scale_factor=0, energy=slab_E)

        slab_Area = slab_obj.surface_area

        # Formulas
        oriented_formula = oriented_struct.composition.reduced_formula
        slab_formula = slab_struct.composition.reduced_formula

        # Compositions
        bulk_comp = oriented_struct.composition.as_dict()
        slab_comp = slab_struct.composition.as_dict()

        bulk_unit_form_dict = Composition({el: bulk_comp[el] for el in bulk_comp.keys() if el != "O"}).as_dict()
        slab_unit_form_dict = Composition({el: slab_comp[el] for el in bulk_comp.keys() if el != "O"}).as_dict()

        bulk_unit_form = sum(bulk_unit_form_dict.values())
        slab_unit_form = sum(slab_unit_form_dict.values())
        slab_bulk_ratio = slab_unit_form / bulk_unit_form

        # Surface energy for non-dipolar, symmetric and stoichiometric
        if not slab_obj.is_polar() and slab_obj.is_symmetric() and slab_formula == oriented_formula:
            surface_energy = self.get_surface_energy(slab_E, oriented_E, slab_bulk_ratio, slab_Area)

        else:
            surface_energy = None

        # TODO: Surface energy for non-stochiometric

        # Summary dict
        summary_dict["oriented_struct"] = oriented_struct.as_dict()
        summary_dict["slab_struct"] = slab_struct.as_dict()
        summary_dict["oriented_E"] = oriented_E
        summary_dict["slab_E"] = slab_E
        summary_dict["slab_Area"] = slab_Area
        summary_dict["is_polar"] = str(slab_obj.is_polar())
        summary_dict["is_symmetric"] = str(slab_obj.is_symmetric())
        if slab_formula == oriented_formula:
            summary_dict['is_stoichiometric'] = str(True)
        else:
            summary_dict['is_stoichiometric'] = str(False)

        summary_dict['N'] = slab_bulk_ratio
        summary_dict['surface_energy'] = surface_energy

        # Add results to db
        if to_db:
            mmdb.collection = mmdb.db["surface_energies"]
            mmdb.collection.insert_one(summary_dict)

        else:
            with open("{}_{}_surface_energy.json".format(slab_formula, miller_index), "w") as f:
                f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # Logger
        logger.info("{}_{} Surface Energy: {} [eV/A**2]".format(slab_formula, miller_index, surface_energy))

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
        gamma_hkl = (slab_E - (slab_bulk_ratio * oriented_E)) / (2*slab_Area) #scaling for bulk!
        return gamma_hkl


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
    required_params = ['bulk_structure', 'db_file']
    optional_params = ['wulff_plot', 'to_db']

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", False)
        wulff_plot = self.get("wulff_plot", True)
        bulk_structure = self["bulk_structure"]
        Ev2Joule = 16.0219 # eV/Angs2 to J/m2
        summary_dict = {}

        # Bulk formula
        bulk_formula = bulk_structure.composition.reduced_formula

        # Connect to DB and Surface Energies collection
        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
        collection = mmdb.db["surface_energies"]

        # Find Surface energies for the given material + facet
        docs = collection.find({"task_label": {"$regex": "^{}".format(bulk_formula)}})

        # Surface energy and structures dictionary
        surface_energies_dict = {}
        structures_dict = {}
        for d in docs:
            slab_struct = d["slab_struct"] #as dict
            miller_index = tuple(map(int, d["miller_index"]))
            surface_energy = abs(round(d["surface_energy"] * Ev2Joule, 4)) #Round to 4 decimals
            surface_energies_dict.update({miller_index: surface_energy})
            structures_dict.update({d["miller_index"]: slab_struct})

        # Wulff Analysis
        wulffshape_obj, wulff_info, area_frac_dict = self.get_wulff_analysis(bulk_structure, surface_energies_dict)

        # Store data on summary_dict
        summary_dict['task_label'] = "{}_wulff_shape".format(bulk_formula)
        summary_dict['surface_enegies'] = json_format(surface_energies_dict)
        summary_dict['wulff_info'] = wulff_info
        summary_dict['area_fractions'] = area_frac_dict
        summary_dict['slab_structures'] = structures_dict

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
        from pymatgen.analysis.wulff import WulffShape

        # Get key/values from energies dict J/m^2
        miller_list = surface_energies_dict.keys()
        e_surf_list = surface_energies_dict.values()

        # WulffShape Analysis
        wulffshape_obj = WulffShape(bulk_structure.lattice, miller_list, e_surf_list)

        # Collect wulffshape properties
        shape_factor = wulffshape_obj.shape_factor
        anisotropy = wulffshape_obj.anisotropy
        weight_surf_energy = wulffshape_obj.weighted_surface_energy # J/m2
        shape_volume =  wulffshape_obj.volume
        effective_radius = wulffshape_obj.effective_radius
        area_frac_dict = json_format(wulffshape_obj.area_fraction_dict) # {hkl: area_hkl/total area on wulff}

        wulff_info = {'shape_factor': shape_factor, 'anisotropy': anisotropy,
                      'weight_surf_energy': weight_surf_energy, 'volume': shape_volume,
                      'effective_radius': effective_radius}

        return wulffshape_obj, wulff_info, area_frac_dict
