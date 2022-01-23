import unittest
import json
from pymongo import MongoClient


class RegressionTest(unittest.TestCase):
    def setUp(self):
        # Connect to the local fireworks db
        client = MongoClient("mongodb://localhost:27017")
        self.db = client.get_database("fireworks")
        # Read the JSON hash from executing the workflow
        # Note that these hashes are specific to the github host runner
        with open("db_hashes.json") as f:
            self.hash_dict = json.load(f)
        self.db_hash = "983357eed593c66edea3174911effcd9"
        self.IrO2_101_Ir_oer_single_site_hash = "306ec6114c3f92814ae1dcbf0f36b430"
        self.IrO2_110_Ir_oer_single_site_hash = "d4bbded9c8aaa3754ba0f46a6f2dd04d"
        self.fireworks_hash = "21e09ce2884dc8118bb8a43816d176c9"

    def test_db_hashes(self):
        """This checks that the output of the
        IrO2 workflow matches the one that we
        sucessfully froze. Any deviation will
        be reflected in a different hash for
        the fireworks db and for a collection.
        """
        # Checks the entire DB hash (all collections)
        self.assertEqual(self.db_hash, self.hash_dict["md5"])
        # Checks the 101 single site collection
        self.assertEqual(
            self.IrO2_101_Ir_oer_single_site_hash,
            self.hash_dict["collections"]["IrO2-101_Ir_oer_single_site"],
        )
        # Checks the fireworks collection (there should be a total of 50)
        self.assertEqual(
            self.fireworks_hash, self.hash_dict["collections"]["fireworks"]
        )

    def test_surface_energies(self):
        # Connect to the surface energy collection
        # Get 110 surface energy
        self.IrO2_110_surface_energy = self.db.surface_energies.find_one(
            {"task_label": {"$regex": "110"}}, {"surface_energy": 1}
        )["surface_energy"]
        # Get 101 surface energy
        self.IrO2_101_surface_energy = self.db.surface_energies.find_one(
            {"task_label": {"$regex": "101"}}, {"surface_energy": 1}
        )["surface_energy"]
        self.assertAlmostEqual(self.IrO2_110_surface_energy, 0.08351954123328009)
        self.assertAlmostEqual(self.IrO2_101_surface_energy, 0.09809758578479011)
