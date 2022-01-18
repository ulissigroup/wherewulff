from typing import Counter
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure, env_chk
from atomate.vasp.firetasks.run_calc import RunVaspFake

from CatFlows.dft_settings.settings import MOSurfaceSet
from CatFlows.firetasks.handlers import ContinueOptimizeFW


# Dictionary that holds the paths to the VASP input
# and output files. Right now this assumes that the code is run
# in a container (/home/jovyan) with the files placed in the right folder.
# Maps fw_name to the ref_dir
ref_dirs = {
    "RuO2_110 bulk optimization": "/home/jovyan/mo-wflow-new/RuO2_bulk_110",
    "RuO2_101 bulk optimization": "/home/jovyan/mo-wflow-new/RuO2_bulk_101",
    "RuO2_110 slab optimization": "/home/jovyan/mo-wflow-new/RuO2_slab_110",
    "RuO2_101 slab optimization": "/home/jovyan/mo-wflow-new/RuO2_slab_101",
    "IrO2_110 bulk optimization": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_bulk_110",
    "IrO2_101 bulk optimization": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_bulk_101",
    "IrO2_110 slab optimization": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_slab_110",
    "IrO2_101 slab optimization": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_slab_101",
    "IrO2-110-O_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_Ox_pbx_1",
    "IrO2-110-OH_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_pbx_1",
    "IrO2-110-OH_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_pbx_2",
    "IrO2-110-OH_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_pbx_3",
    "IrO2-110-OH_4": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_pbx_4",
    "IrO2-101-O_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_Ox_pbx_1",
    "IrO2-101-OH_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_pbx_1",
    "IrO2-101-OH_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_pbx_2",
    "IrO2-101-OH_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_pbx_3",
    "IrO2-101-OH_4": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_pbx_4",
    "IrO2-110-Ir-reference": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_reference",
    "IrO2-110-Ir-OH_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_0",
    "IrO2-110-Ir-OH_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_1",
    "IrO2-110-Ir-OH_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_2",
    "IrO2-110-Ir-OH_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OH_3",
    "IrO2-110-Ir-OOH_up_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_up_0",
    "IrO2-110-Ir-OOH_up_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_up_1",
    "IrO2-110-Ir-OOH_up_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_up_2",
    "IrO2-110-Ir-OOH_up_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_up_3",
    "IrO2-110-Ir-OOH_down_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_down_0",
    "IrO2-110-Ir-OOH_down_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_down_1",
    "IrO2-110-Ir-OOH_down_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_110_OOH_down_2",
    "IrO2-110-Ir-OOH_down_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/XXX",
    "IrO2-101-Ir-reference": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_reference",
    "IrO2-101-Ir-OH_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_0",
    "IrO2-101-Ir-OH_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_1",
    "IrO2-101-Ir-OH_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_2",
    "IrO2-101-Ir-OH_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OH_3",
    "IrO2-101-Ir-OOH_up_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_up_0",
    "IrO2-101-Ir-OOH_up_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_up_1",
    "IrO2-101-Ir-OOH_up_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_up_2",
    "IrO2-101-Ir-OOH_up_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_up_3",
    "IrO2-101-Ir-OOH_down_0": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_down_0",
    "IrO2-101-Ir-OOH_down_1": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_down_1",
    "IrO2-101-Ir-OOH_down_2": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_down_2",
    "IrO2-101-Ir-OOH_down_3": "/home/jovyan/mo-wflow-new/IrO2_full_jh/IrO2_101_OOH_down_3",

}


def Bulk_FW(
    bulk,
    name="",
    vasp_input_set=None,
    parents=None,
    wall_time=172800,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
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
    if not vasp_input_set:
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
    if run_fake:
        assert (
            "RuO2" in name or "IrO2" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": 172800})

    # Append Continue-optimizeFW for wall-time handling and use for uuid message
    # passing
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=True, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add bulk_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_bulk_uuid})

    return fw


def Slab_FW(
    slab,
    name="",
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=172800,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
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
    if not vasp_input_set:
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
    if run_fake:
        assert (
            "RuO2" in name or "IrO2" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_slab_uuid})

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw


def AdsSlab_FW(
    slab,
    name="",
    oriented_uuid="",
    slab_uuid="",
    ads_slab_uuid="",
    is_adslab=True,
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=172800,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a ads_slab firework. Returns an OptimizeFW for the specified slab.

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

    # DFT Method
    if not vasp_input_set:
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
            "uuid": ads_slab_uuid,
            "wall_time": wall_time,
            "name": name,
            "max_tries": 5,
            "is_bulk": False,
            "is_adslab": is_adslab,
            "oriented_uuid": oriented_uuid,  # adslab FW should get terminal node ids
            "slab_uuid": slab_uuid,
            "is_bulk": False,
        },
    )
    if run_fake:
        assert (
            "RuO2" in name or "IrO2" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": ads_slab_uuid})

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw
