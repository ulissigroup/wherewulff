import numpy as np

from pymatgen.analysis.adsorption import AdsorbateSiteFinder


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
