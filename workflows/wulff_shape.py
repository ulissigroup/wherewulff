from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

import pymatgen
from pymatgen.core import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, get_symmetrically_distinct_miller_indices
from pymatgen.io.vasp.sets import MVLSlabSet

from fireworks import Firework, Workflow, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import VASP_CMD, DB_FILE


from atomate.utils.utils import get_meta_from_structure

from fireworks.scripts import lpad_run

from analysis.wulff_shape import WulffShapeFW


def WulffShape_WF(bulk_structure, parents=None, vasp_cmd=VASP_CMD, db_file=DB_FILE):
    """
    Wrap-up workflow to do the Wulff Shape Analysis after MO_SLABS_WF.

    Args:
        bulk_structure (Structure): Bulk structure to refer the wulff shape
        vasp_cmd: vasp executable
        db_file: database file.

    Returns:
        JSON file with Wulff Analysis.
    """
    # Bulk structure formula
    bulk_formula = bulk_structure.composition.reduced_formula

    # WulffShape Analysis
    wulff_fw = Firework(WulffShapeFW(bulk_structure=bulk_structure,
                                     db_file=db_file),
                                     name="{} wulff shape Task".format(bulk_formula),
                                     parents=parents)

    all_fws = [wulff_fw]
    all_fws.extend(parents)
    wulff_wf = Workflow(all_fws, name="{} wulff shape analysis".format(bulk_formula))

    return wulff_wf