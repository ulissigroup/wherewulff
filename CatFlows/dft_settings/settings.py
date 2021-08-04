import numpy as np

from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.io.vasp.inputs import Kpoints


# Theoretical Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """
    def __init__(self, structure, psp_version="PBE_54", bulk=False, auto_dipole=True, **kwargs):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, **kwargs)
        
        #self.structure = structure
        self.psp_version = psp_version
        self.bulk = bulk
        self.auto_dipole = auto_dipole

        # Change the default PBE version from Pymatgen
        psp_versions = ['PBE', 'PBE_52', 'PBE_54']
        assert self.psp_version in psp_versions 
        MOSurfaceSet.CONFIG['POTCAR_FUNCTIONAL'] = self.psp_version


    # Setting magmom automatically
    def set_magmom(self):
        """ Returns a list of magnetic moments sorted as site.specie.name traversed """
        magmoms = []
        for site in self.structure:
            element = site.specie.name
            if element == 'O':
                magmoms.append(0.6)
            elif element == 'H':
                magmoms.append(0.1)
            else: # Crystal field ?
                magmoms.append(5.0)
        return magmoms

    # Dipolar moment correction
    def _get_center_of_mass(self):
        """ From coordinates, weighted by specie, Return center of mass """
        weights = [s.species.weight for s in self.structure]
        center_of_mass = np.average(self.structure.frac_coords, weights=weights, axis=0)
        return list(center_of_mass)

    @property
    def incar(self):
        incar = super(MOSurfaceSet, self).incar

        # Direct of reciprocal (depending if its bulk or slab)
        if self.bulk:
            incar["LREAL"] = False
        else:
            incar["LREAL"] = False

        # Setting Magnetic Moments  
        #magmoms = self.set_magmom()
        #incar['MAGMOM'] = magmoms

        # Setting auto_dipole correction (for slabs only)
        if not self.bulk and self.auto_dipole:
            incar["LDIPOL"] = True
            incar["IDIPOL"] = 3
            incar["DIPOL"] = self._get_center_of_mass()

        # Incar Settings for optimization
        incar_config = {"GGA": "PE", "ENCUT": 500, "EDIFF": 1e-5, "EDIFFG": -0.05, 
                        "ISYM": 0, "ISPIN": 2, "ISIF": 0}
        # Update incar
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
