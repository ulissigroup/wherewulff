from __future__ import absolute_import, division, print_function, unicode_literals
import uuid

from fireworks import Firework, Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.firetasks.slab_ads import get_clockwise_rotations
from CatFlows.fireworks.optimize import AdsSlab_FW
from CatFlows.fireworks.surface_pourbaix import SurfacePBX_FW
from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator


def SurfacePBX_WF(slab, slab_uuid, oriented_uuid, adsorbates, vasp_cmd=VASP_CMD, db_dile=DB_FILE):
    """
    Wrap-up Workflow for surface-OH/Ox terminated + SurfacePBX Analysis.

    Args:

    Retruns:
        something
    """
    # Empty list of fws
    hkl_fws, hkl_uuids = [], []

    # Reduced formula and Miller_index
    reduced_formula = slab.composition.reduced_formula
    slab_miller_index = "".join(list(map(str, slab.miller_index)))

    # Generate a set of OptimizeFW additons that will relax all the adslab in parallel
    for adsorbate in adsorbates:
        adslabs = get_clockwise_rotations(slab, adsorbate)
        for adslab_label, adslab in adslabs.items():
            name = f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
            ads_slab_uuid = uuid.uuid4()
            ads_slab_fw = AdsSlab_FW(
                adslab,
                name=name,
                oriented_uuid=oriented_uuid,
                slab_uuid=slab_uuid,
                ads_slab_uuid=ads_slab_uuid,
                vasp_cmd=vasp_cmd
            )
            hkl_fws.append(ads_slab_fw)
            hkl_uuids.append(ads_slab_uuid)

    # Surface PBX Diagram for each surface orientation
    pbx_name = f"Surface-PBX-{slab.composition.reduced_formula}-{slab_miller_index}"
    pbx_fw = SurfacePBX_FW(
             reduced_formula=reduced_formula,
             name=pbx_name,
             miller_index=slab_miller_index,
             slab_uuid=slab_uuid,
             ads_slab_uuids=hkl_uuids,
             parents=hkl_fws,
    )

    # Create the workflow
    all_fws = hkl_fws + [pbx_fw]
    pbx_wf = Workflow(all_fws, name=f"{slab.composition.reduced_formula}-{slab_miller_index}-PBX Workflow")
    return pbx_wf




