from __future__ import absolute_import, division, print_function, unicode_literals
import uuid
import numpy as np

from pymatgen.core.periodic_table import Element

from fireworks import Workflow
from atomate.vasp.config import VASP_CMD, DB_FILE

from CatFlows.fireworks.optimize import AdsSlab_FW
from CatFlows.fireworks.surface_pourbaix import SurfacePBX_FW
from CatFlows.adsorption.MXide_adsorption import MXideAdsorbateGenerator

from CatFlows.workflows.oer import OER_WF


# Angles list
def get_angles(n_rotations=4):
    """Get angles like in the past"""
    angles = []
    for i in range(n_rotations):
        deg = (2 * np.pi / n_rotations) * i
        angles.append(deg)
    return angles


def add_adsorbates(adslab, ads_coords, molecule, z_offset=[0, 0, 0.15]):
    """Add molecule in all ads_coords once"""
    translated_molecule = molecule.copy()
    for ads_site in ads_coords:
        for mol_site in translated_molecule:
            new_coord = ads_site + (mol_site.coords - z_offset)
            adslab.append(
                mol_site.specie,
                new_coord,
                coords_are_cartesian=True,
                properties=mol_site.properties,
            )
    return adslab


# Try the clockwise thing again...
def get_clockwise_rotations(slab_ref, slab, molecule):
    """We need to rush function..."""
    # This will be a inner method
    mxidegen = MXideAdsorbateGenerator(
        slab_ref,
        repeat=[1, 1, 1],
        verbose=False,
        positions=["MX_adsites"],
        relax_tol=0.025,
    )

    # Getting the bulk-like adsites on the original slab
    bulk_like, _ = mxidegen.get_bulk_like_adsites()
    # bulk_like_sites = mxidegen._filter_clashed_sites(bulk_like)  # is needed?

    # Bondlength and X
    # bondlength, X = mxidegen.bondlength, mxidegen.X
    # bulk_like_shifted = _bulk_like_adsites_perturbation(
    #    slab_ref, slab, bulk_like_sites, bondlength=bondlength, X=X
    # )

    _, X = mxidegen.bondlength, mxidegen.X
    bulk_like_shifted = _bulk_like_adsites_perturbation(slab_ref, slab, bulk_like, X=X)

    # set n_rotations to 1 if mono-atomic
    n = len(molecule[0]) if type(molecule).__name__ == "list" else len(molecule)
    n_rotations = 1 if n == 1 else 4

    # Angles
    angles = get_angles(n_rotations=n_rotations)

    # Molecule formula
    molecule_comp = molecule.composition.as_dict()
    molecule_formula = "".join(molecule_comp.keys())

    # rotate OH
    molecule_rotations = mxidegen.get_transformed_molecule_MXides(
        molecule, axis=[0, 0, 1], angles_list=angles
    )

    # placement
    adslab_dict = {}
    for rot_idx in range(len(molecule_rotations)):
        slab_ads = slab.copy()
        slab_ads = add_adsorbates(
            slab_ads, bulk_like_shifted, molecule_rotations[rot_idx]
        )
        slab_ads.sort()  # Sorting for input/output consistency
        adslab_dict.update({"{}_{}".format(molecule_formula, rot_idx + 1): slab_ads})

    return adslab_dict, bulk_like_shifted


def _bulk_like_adsites_perturbation_bondlength(slab_ref, slab, bulk_like_sites, bondlength, X):
    """Let's perturb bulk_like_sites with delta (x,y,z) comparing input and output"""
    slab_ref_coords = slab_ref.cart_coords
    slab_coords = slab.cart_coords

    delta_coords = slab_coords - slab_ref_coords

    metal_idx = []
    for bulk_like_site in bulk_like_sites:
        for idx, site in enumerate(slab_ref):
            if (
                site.specie != Element(X)
                and site.coords[2] > slab_ref.center_of_mass[2]
            ):
                dist = np.linalg.norm(bulk_like_site - site.coords)
                if dist < bondlength:
                    metal_idx.append(idx)

    bulk_like_deltas = [delta_coords[i] for i in metal_idx]
    return [n + m for n, m in zip(bulk_like_sites, bulk_like_deltas)]

def _bulk_like_adsites_perturbation(slab_ref, slab, bulk_like_sites, X):
    """Let's perturb bulk_like_sites with delta (x,y,z) comparing input and output"""
    slab_ref_coords = slab_ref.cart_coords
    slab_coords = slab.cart_coords

    delta_coords = slab_coords - slab_ref_coords

    metal_idx = []
    for bulk_like_site in bulk_like_sites:
        min_dist = np.inf # intialize min_dist register
        min_metal_idx = 0
        end_idx = np.where(slab_ref.frac_coords[:,2] >= slab_ref.center_of_mass[2])[0][-1]
        # FIXME: I think we can make this faster by replacing with a while loop
        # and only looping over the top half
        for idx, site in enumerate(slab_ref):
            if (
                site.specie != Element(X)
                and site.frac_coords[2]
                > slab_ref.center_of_mass[2] # go over the top half of slab
                ):
                dist = np.linalg.norm(bulk_like_site - site.coords)
                
                if dist < min_dist:
                    min_dist = dist
                    min_metal_idx = idx
                    
            if idx == end_idx:
                metal_idx.append(min_metal_idx)

    bulk_like_deltas = [delta_coords[i] for i in metal_idx]
    return [n + m for n, m in zip(bulk_like_sites, bulk_like_deltas)]


def SurfacePBX_WF(
    bulk_structure,
    slab,
    slab_orig,
    slab_uuid,
    oriented_uuid,
    adsorbates,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    metal_site="",
    applied_potential=1.60,
    applied_pH=0.0,
):
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
    ads_slab_orig = {}
    for adsorbate in adsorbates:
        adslabs, bulk_like_shifted = get_clockwise_rotations(slab_orig, slab, adsorbate)
        for adslab_label, adslab in adslabs.items():
            name = (
                f"{slab.composition.reduced_formula}-{slab_miller_index}-{adslab_label}"
            )
            ads_slab_uuid = str(uuid.uuid4())
            ads_slab_fw = AdsSlab_FW(
                adslab,
                name=name,
                oriented_uuid=oriented_uuid,
                slab_uuid=slab_uuid,
                ads_slab_uuid=ads_slab_uuid,
                vasp_cmd=vasp_cmd,
                db_file=db_file,
            )
            ads_slab_orig.update({adslab_label: adslab})
            hkl_fws.append(ads_slab_fw)
            hkl_uuids.append(ads_slab_uuid)

    # Surface PBX Diagram for each surface orientation
    surface_pbx_uuid = str(uuid.uuid4())
    pbx_name = f"Surface-PBX-{slab.composition.reduced_formula}-{slab_miller_index}"
    pbx_fw = SurfacePBX_FW(
        reduced_formula=reduced_formula,
        name=pbx_name,
        miller_index=slab_miller_index,
        slab_uuid=slab_uuid,
        oriented_uuid=oriented_uuid,
        ads_slab_uuids=hkl_uuids,
        parents=hkl_fws,
        surface_pbx_uuid=surface_pbx_uuid,
    )

    # OER workflow
    oer_fw = OER_WF(
        bulk_structure=bulk_structure,
        miller_index=slab_miller_index,
        slab_orig=slab_orig,
        bulk_like_sites=bulk_like_shifted,
        ads_dict_orig=ads_slab_orig,
        metal_site=metal_site,
        applied_potential=applied_potential,
        applied_pH=applied_pH,
        parents=[pbx_fw],
        vasp_cmd=VASP_CMD,
        db_file=DB_FILE,
        surface_pbx_uuid=surface_pbx_uuid,
    )

    # Create the workflow
    all_fws = hkl_fws + [pbx_fw] + [oer_fw]
    oer_wf = Workflow(
        all_fws,
        name=f"{slab.composition.reduced_formula}-{slab_miller_index}-PBX Workflow",
    )
    return oer_wf
