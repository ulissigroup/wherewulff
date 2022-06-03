from os import sched_get_priority_max
import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.io.vasp.inputs import Kpoints


# Get bulk initial magnetic moments
def set_bulk_magmoms(structure, nm_magmom_buffer=0.6, tol=0.1, scale_factor=1.2):
    """
    Returns decorated bulk structure with initial magnetic moments,
    based on crystal-field theory for TM.
    """
    struct = structure.copy()
    # Voronoi NN
    voronoi_nn = VoronoiNN(tol=tol)
    # SPG Analysis
    sga = SpacegroupAnalyzer(struct)
    sym_struct = sga.get_symmetrized_structure()
    # Magnetic moments
    element_magmom = {}
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        if site.specie.is_transition_metal:
            cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
            cn = round(cn, 5)
            # Filter between Oh or Td Coordinations
            if cn > 5.0:
                coordination = "oct"
            else:
                coordination = "tet"
            # Spin configuration depending on row
            if site.specie.row >= 5.0:
                spin_config = "low"
            else:
                spin_config = "high"
            # Magnetic moment per metal site
            magmom = site.specie.get_crystal_field_spin(
                coordination=coordination, spin_config=spin_config
            )
            # Add to dict
            element_magmom.update(
                {
                    str(site.species_string): abs(scale_factor * float(magmom))
                    if magmom > 0.01
                    else nm_magmom_buffer
                }
            )

        elif site.specie.is_chalcogen:  # O
            element_magmom.update({str(site.species_string): 0.6})

        else:
            element_magmom.update({str(site.species_string): nm_magmom_buffer})

    magmoms = [element_magmom[site.species_string] for site in struct]

    # Decorate
    for site, magmom in zip(struct.sites, magmoms):
        site.properties["magmom"] = round(magmom, 2)
    return struct


class SelectiveDynamics(AdsorbateSiteFinder):
    """
    Different methods for Selective Dynamics.
    """

    def __init__(self, slab):
        self.slab = slab.copy()

    @classmethod
    def center_of_mass(cls, slab):
        """Method based of center of mass."""
        sd_list = []
        sd_list = [
            [False, False, False]
            if site.frac_coords[2] < slab.center_of_mass[2]
            else [True, True, True]
            for site in slab.sites
        ]
        new_sp = slab.site_properties
        new_sp["selective_dynamics"] = sd_list
        return slab.copy(site_properties=new_sp)


# Theoretical DFT Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """

    def __init__(
        self,
        structure,
        psp_version="PBE_54",
        bulk=False,
        set_mix=False,
        auto_dipole=True,
        initial_magmoms=None,
        apply_U=False,
        UL=None,
        UJ=None,
        UU=None,
        **kwargs
    ):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, set_mix=False, **kwargs
        )

        self.apply_U = apply_U
        self.UL = UL
        self.UJ = UJ
        self.UU = UU
        # self.structure = structure
        self.psp_version = psp_version
        self.bulk = bulk
        self.auto_dipole = auto_dipole
        self.initial_magmoms = initial_magmoms
        self.set_mix = set_mix

        # Change the default PBE version from Pymatgen
        psp_versions = ["PBE", "PBE_52", "PBE_54"]
        assert self.psp_version in psp_versions
        # MOSurfaceSet.CONFIG["POTCAR_FUNCTIONAL"] = self.psp_version
        self.potcar_functional = self.psp_version

    # Dipolar moment correction
    def _get_center_of_mass(self):
        """From coordinates, weighted by specie, Return center of mass"""
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
            incar["LREAL"] = True

        # Setting auto_dipole correction (for slabs only)
        #       if not self.bulk and self.auto_dipole:
        #           incar["LDIPOL"] = True
        #           incar["IDIPOL"] = 3
        #           incar["DIPOL"] = self._get_center_of_mass()

        # Setting magnetic moments for children
        if self.initial_magmoms:
            incar["MAGMOM"] = self.initial_magmoms

        if "LDAUPRINT" in incar.keys():
            incar["LDAUPRINT"] = 0  # silent mode

        # Setting the Hubbard and Hund params if any
        if self.apply_U:
            incar["LDAU"] = True
            incar["LDAUL"] = self.UL
            incar["LDAUJ"] = self.UJ
            incar["LDAUU"] = self.UU

        # Incar Settings for optimization
        incar_config = {
            "GGA": "PE",
            "ENCUT": 500,
            "EDIFF": 1e-4,
            "EDIFFG": -0.05,
            "ISYM": 0,
            "SYMPREC": 1e-10,
            "ISPIN": 2,
            "ISIF": 0,
            "NSW": 300,
            "NCORE": 4,
            "LWAVE": True,
            "LCHARG": False,
            "LVTOT": False,
            "ISTART": 1,
            "NELM": 80,
            "NCORE": 4,
            "ISMEAR": 0,
            "SIGMA": 0.2,
        }
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
            kpts = tuple(np.ceil(50.0 / abc).astype("int"))
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0, 0, 0))

        else:
            kpts = np.ceil(30.0 / abc).astype("int")
            kpts[2] = 1
            kpts = tuple(kpts)
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0, 0, 0))
