import os
import json
from datetime import datetime

from pydash.objects import has, get

import pymatgen
from pymatgen import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator, generate_all_slabs, get_symmetrically_distinct_miller_indices
from pymatgen.io.vasp.sets import MVLSlabSet


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

# Theoretical Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """
    def __init__(self, structure, bulk=False, **kwargs):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, **kwargs)

        # Bulks are better with LREAL=.FALSE.
        if bulk:
            self._config_dict['INCAR'].update({'LREAL': False})
        else:
            self._config_dict['INCAR'].update({'LREAL': True})

    @property
    def incar(self):
        incar = super(MOSurfaceSet, self).incar

        #Setting Magnetic Moments  
        magmom = set_magmom(self.structure)

        # Incar Settings for optimization
        incar_config = {"ENCUT": 500, "EDIFF": 1e-5, "EDIFFG": -0.05,
                        "POTIM": 1.0, "ISYM": 0, "ISPIN": 2, "ISIF": 0,
                        "MAGMOM": magmom}
        #update incar
        incar.update(incar_config)
        incar.update(self.user_incar_settings)

        return incar

@explicit_serialize
class SurfaceEnergy(FiretaskBase):
    """
    Computes the surface energy for stoichiometric slab models.

    Args:
        tag: datetime string for folder name and id.
        db_file: database file path
        slab_formula: Reduced formula of the slab model e.g (RuO2)
        miller_index: Crystallographic orientations of the slab model.
        to_db (default: True): Save the data on the db or in a json_file.

    return:
        summary_dict with surface energy information.
    """
    required_params = ['tag', 'db_file', 'slab_formula', 'miller_index']
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        tag = self['tag']
        db_file = env_chk(self.get("db_file"), fw_spec)
        slab_formula = self["slab_formula"]
        miller_index = self["miller_index"]
        to_db = self.get("to_db", True)
        summary_dict = {"slab_formula": slab_formula, "miller_index": miller_index}

        # Collect and store tasks_ids
        all_task_ids = []

        mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

        oriented = mmdb.collection.find_one({'task_label': '{}_{}_{} oriented bulk optimization'.format(slab_formula, miller_index, tag)})
        slab = mmdb.collection.find_one({'task_label': '{}_{}_{} slab optimization'.format(slab_formula, miller_index, tag)})

        all_task_ids.append(oriented['task_id'])
        all_task_ids.append(slab['task_id'])

        # Get Structures from DB
        oriented_struct = Structure.from_dict(oriented["calcs_reversed"][-1]["output"]['structure'])
        slab_struct = Structure.from_dict(slab["calcs_reversed"][-1]["output"]['structure'])

        # Get DFT Energies from DB
        oriented_E = oriented["calcs_reversed"][-1]["output"]["energy"]
        slab_E = slab["calcs_reversed"][-1]["output"]["energy"]

        slab_Area = slab_struct.surface_area

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
        scale_factor = slab_unit_form / bulk_unit_form

        # Surface energy for non-dipolar, symmetric and stoichiometric
        if not slab_struct.is_polar() and slab_struct.is_symmetric() and slab_formula == oriented_formula:
            surface_energy = get_surface_energy(slab_E, oriented_E, scale_factor, slab_area)

        else:
            surface_energy = None

        # TODO: Surface energy for non-stochiometric

        # Summary dict
        summary_dict["oriented_struct"] = oriented_struct.as_dict()
        summary_dict["slab_struct"] = slab_struct.as_dict()
        summary_dict["oriented_E"] = oriented_E
        summary_dict["slab_E"] = slab_E
        summary_dict["slab_Area"] = slab_Area
        summary_dict["is_polar"] = slab_struct.is_polar()
        summary_dict["is_symmetric"] = slab_struct.is_symmetric()
        if slab_formula == oriented_formula:
            summary_dict['is_stoichiometric'] = True
        else:
            summary_dict['is_stoichiometric'] = False

        summary_dict['N'] = scale_factor
        summary_dict['surface_energy'] = surface_energy

        # Add results to db
        if to_db:
            mmdb.collection = mmdb.db["{}_{} surface_energy".format(slab_formula, miller_index)]
            mmdb.collection.insert_one(summary_dict)

        else:
            with open("{}_{}_surface_energy.json".format(slab_formula, miller_index), "W") as f:
                f.write(json.dumps(summary_dict, default=DATETIME_HANDLER))

        # Logger
        logger.info("{}_{} Surface Energy: {} [eV/A**2]".format(slab_formula, miller_index, surface_energy))

    def get_surface_energy(slab_E, oriented_E, scale_factor, slab_Area):
        """
        Surface energy for non-dipolar, symmetric and stoichiometric
        Units: eV/A**2

        Args:
            slab_E: DFT energy from slab optimization [eV]
            oriented_E: DFT energy from oriented bulk optimization [eV]
            scale_factor: slab units formula per bulk units formula 
            slab_area: Area from the slab model XY plane [A**2]
        Return:
            gamma_hkl - Surface energy for symmetric and stoichiometric model. 
        """
        gamma_hkl = (slab_E - (scale_factor * oriented_E)) / (2*slab_Area) #scaling for bulk!
        return gamma_hkl



#TODO: First try some materiales to see how it does.

# # Wulff Construction Method
# @explicit_serialize
# class WulffAnalysis(FiretaskBase):
#     """
#     Custom Atomate Analysis task to build the Wulff construction
#     of Metal Oxide materials.

#     Args:
#         tag:
#         db_file:

#     """
#     required_params = []
#     optional_params = []

#     def run_task(self, fw_spec):
#         from pymatgen.analysis.wulff import WulffShape

#         # Bulk Lattice

#         # Function to compute surface energy returning the following dict

#         # Surface energy values in J/m^2
#         surface_energies = {miller_index: energy}
#         miller_list = surface_energies.keys()
#         e_surf_list = surface_energies.values()

#         # Construct a Wulff shape
#         wulffshape = WulffShape(bulk.lattice, miller_list, e_surf_list)

#         # Get info from WulffShape - slab contribution - image

#         # Export JSON file and/or filter non-contribution slabs for Reactivity
