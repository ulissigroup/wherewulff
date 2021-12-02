import json
import uuid

import numpy as np

from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import Slab

from pydash.objects import has, get

from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk
from atomate.utils.utils import get_logger
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)


@explicit_serialize
class OER_SingleSiteAnalyzer(FiretaskBase):
    """
    Post-processing FireTask to derive Delta G's for
    OER (WNA) mechanism in a single active site and
    derive the theoretical overpotential.

    Args:


    Returns:
        Reactivity post-processing for a given surface
        and DB json data.
    """

    required_params = ["db_file"]
    optional_params = ["to_db"]

    def run_task(self, fw_spec):

        # Variables
        db_file = env_chk(self.get("db_file"), fw_spec)
        to_db = self.get("to_db", True)
