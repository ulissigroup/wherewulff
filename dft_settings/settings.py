import os
import json
from datetime import datetime
import numpy as np

from pydash.objects import has, get

import pymatgen
from pymatgen.core import Structure, Lattice
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


# Theoretical Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """
    def __init__(self, structure, psp_version="PBE_54", bulk=False, **kwargs):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, **kwargs)
        

        self.psp_version = psp_version

        # Change the default PBE version from Pymatgen
        psp_versions = ['PBE', 'PBE_52', 'PBE_54']
        assert self.psp_version in psp_versions 
        MOSurfaceSet.CONFIG['POTCAR_FUNCTIONAL'] = self.psp_version

        # Bulks are better with LREAL=.FALSE. ??
        if bulk:
            self._config_dict['INCAR'].update({'LREAL': False})
        else:
            self._config_dict['INCAR'].update({'LREAL': True})

    # Setting magmom automatically
    def set_magmom(self, structure):
        """Expects a pymatgen structure object
        and returns a list of magmoms in the same order
        as the site.specie.name traversed"""

        magmoms = []
        for site in self.structure:
            element = site.specie.name
            if element == 'O':
                magmoms.append(0.6)
            elif element == 'H':
                magmoms.append(0.1)
            else: # M Transition metal ?
                magmoms.append(5.0)

        return magmoms

    @property
    def incar(self):
        incar = super(MOSurfaceSet, self).incar

        #Setting Magnetic Moments  
        magmoms = self.set_magmom(self.structure)
        incar['MAGMOM'] = magmoms

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
