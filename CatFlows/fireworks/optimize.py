from typing import Counter
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure

from CatFlows.dft_settings.settings import MOSurfaceSet
from CatFlows.firetasks.handlers import ContinueOptimizeFW


def Bulk_FW(
    bulk,
    name="",
    parents=None,
    wall_time=172800,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Function to generate a bulk firework. Returns an OptimizeFW for the specified slab.

    Args:
        bulk              (Struct Object)   : Structure corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 172800) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to bulk calculation.
    """
    import uuid

    # Generate a unique ID for Bulk_FW
    fw_bulk_uuid = uuid.uuid4()

    # DFT Method
    vasp_input_set = MOSurfaceSet(bulk, bulk=True)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=bulk,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "_pass_job_info": True,
            "uuid": fw_bulk_uuid,
            "wall_time": wall_time,
            "max_tries": 5,
            "name": name,
            "is_bulk": True,
        },
    )
    # Switch-off GzipDir for WAVECAR transferring
    fw.tasks[1].update({"gzip_output": False})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(ContinueOptimizeFW(is_bulk=True, counter=0, vasp_cmd=vasp_cmd))

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_bulk_uuid})

    # Switch-on WalltimeHandler in RunVaspCustodian
    if wall_time is not None:
        fw.tasks[1].update({"wall_time": wall_time})

    return fw


def Slab_FW(
    slab,
    name="",
    parents=None,
    add_slab_metadata=True,
    wall_time=172800,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Function to generate a slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)     : Slab corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 172800) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """
    import uuid

    # Generate a unique ID for Slab_FW
    fw_slab_uuid = uuid.uuid4()

    # DFT Method
    vasp_input_set = MOSurfaceSet(slab, bulk=False)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=slab,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "_pass_job_info": True,
            "uuid": fw_slab_uuid,
            "wall_time": wall_time,
            "max_tries": 5,
            "name": name,
            "is_bulk": False,
        },
    )
    # Switch-off GzipDir for WAVECAR transferring
    fw.tasks[1].update({"gzip_output": False})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(ContinueOptimizeFW(is_bulk=False, counter=0, vasp_cmd=vasp_cmd))

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_slab_uuid})

    # Switch-on WalltimeHandler in RunVaspCustodian
    if wall_time is not None:
        fw.tasks[1].update({"wall_time": wall_time})

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[-1]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw
