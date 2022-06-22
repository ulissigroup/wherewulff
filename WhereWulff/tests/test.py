import os
import json
import shutil
import warnings
import unittest

import numpy as np

from pymatgen.core.structure import Structure
from WhereWulff.workflows.catflows import CatFlows

warnings.filterwarnings("ignore")


class CatFlowsTest(unittest.TestCase):
    """Unittest for the General Workflow"""

    def setUp(self):
        """Automatically called for every single test"""
        self.cif_file = "./RuO2_136.cif"
        self.catflows = CatFlows(self.cif_file)

        self.bulk_formula = self.catflows.bulk_formula
        self.max_index = self.catflows.max_index
        return

    def test_magmoms(self):
        """Check magnetic moments"""
        struct_mag = self.catflows._get_bulk_magmoms()
        magmoms = struct_mag.site_properties["magmom"]
        return self.assertEqual(magmoms, [2.4, 2.4, 0.6, 0.6, 0.6, 0.6])

    def test_bulk_formula(self):
        """Check Bulk reduced formula"""
        bulk_formula = self.catflows._get_bulk_formula()
        return self.assertEqual(bulk_formula, "RuO2")

    def test_miller_indices(self):
        """Check miller indices list"""
        miller_indices = self.catflows._get_miller_indices()
        max_index = np.unique([np.max(hkl) for hkl in miller_indices])[0]
        return self.assertEqual(max_index, self.max_index)

    def test_slab_structures(self):
        """Check Slab structures"""
        slab_structures = self.catflows._get_slab_structures()
        for slab in slab_structures:
            slab_formula = slab.composition.reduced_formula
            if (
                not slab.is_polar()
                and slab.is_symmetric()
                and slab_formula == self.bulk_formula
            ):
                filter = True
        return self.assertTrue(filter)


# Execute Testing
if __name__ == "__main__":
    unittest.main()
