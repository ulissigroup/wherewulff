import numpy as np
from pymatgen.core.structure import Molecule
from pymatgen.core import Structure


"""
Quick manuel enumeration of a few basic molecules for
OER. Rotational degree of freedom handled by function.
"""

# Molecules

# OH and Ox molecules
OH = Molecule(
    ["O", "H"],
    [[0, 0, 0], [-1.0, 0.0, 0.422]],
    site_properties={"magmom": [0.6, 0.1], "binding_site": [True, False]},
)
Ox = Molecule(
    ["O"], [[0, 0, 0]], site_properties={"magmom": [0.6], "binding_site": [True]}
)
OH_Ox_list = [OH, Ox]
OOH_up = Molecule(
    ["O", "O", "H"],
    [[0, 0, 0], [-1.067, -0.403, 0.796], [-0.696, -0.272, 1.706]],
    site_properties={"magmom": [0.6, 0.6, 0.1], "binding_site": [True, False, False]},
)
OOH_down = Molecule(
    ["O", "O", "H"],
    [[0, 0, 0], [-1.067, -0.403, 0.796], [-1.84688848, -0.68892498, 0.25477651]],
    site_properties={"magmom": [0.6, 0.6, 0.1], "binding_site": [True, False, False]},
)
# CH3OH, CH3O, CH2O, CHO adsorbates
#### METHANOL ######
struct_methanol = Structure.from_file("/home/jovyan/wherewulff/WhereWulff/adsorption/POSCAR_methanol_vasp_opt")
# Recenter and make the Oxygen specie that first atom in line
cart_coords = struct_methanol.cart_coords
center_cart_coords = cart_coords - cart_coords[-1]
species = [x.symbol for x in reversed(struct_methanol.species)]
mol_methanol = Molecule(species, np.array([x for x in reversed(center_cart_coords)]))
mol_methanol.add_site_property(
    "binding_site", [True, False, False, False, False, False]
)
mol_methanol.add_site_property("magmom", [0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
#### METHOXY ######
struct_methoxy = Structure.from_file("/home/jovyan/wherewulff/WhereWulff/adsorption/POSCAR_relaxed_tilted_methoxy")
# Recenter and make the Oxygen specie that first atom in line
cart_coords = struct_methoxy.cart_coords
center_cart_coords = cart_coords - cart_coords[-1]
species = [x.symbol for x in reversed(struct_methoxy.species)]
mol_methoxy = Molecule(species, np.array([x for x in reversed(center_cart_coords)]))
mol_methoxy.add_site_property("binding_site", [True, False, False, False, False])
mol_methoxy.add_site_property("magmom", [0.6, 0.1, 0.1, 0.1, 0.1])
#### FORMALDEHYDE ######
struct_formaldehyde = Structure.from_file("/home/jovyan/wherewulff/WhereWulff/adsorption/POSCAR_relaxed_tilted_formaldehyde")
# Recenter and make the Oxygen specie that first atom in line
cart_coords = struct_formaldehyde.cart_coords
center_cart_coords = cart_coords - cart_coords[-1]
species = [x.symbol for x in reversed(struct_formaldehyde.species)]
mol_formaldehyde = Molecule(
    species, np.array([x for x in reversed(center_cart_coords)])
)
mol_formaldehyde.add_site_property("binding_site", [True, False, False, False])
mol_formaldehyde.add_site_property("magmom", [0.6, 0.1, 0.1, 0.1])
#### CHO ######
struct_CHO = Structure.from_file("/home/jovyan/wherewulff/WhereWulff/adsorption/POSCAR_relaxed_tilted_CHO")
# Recenter and make the Oxygen specie that first atom in line
cart_coords = struct_CHO.cart_coords
center_cart_coords = cart_coords - cart_coords[-1]
species = [x.symbol for x in reversed(struct_CHO.species)]
mol_CHO = Molecule(species, np.array([x for x in reversed(center_cart_coords)]))
mol_CHO.add_site_property("binding_site", [True, False, False])
mol_CHO.add_site_property("magmom", [0.6, 0.1, 0.1])

oer_adsorbates_dict = {
    "OH": OH,
    "Ox": Ox,
    "OOH_up": OOH_up,
    "OOH_down": OOH_down,
    "CH3OH": mol_methanol,
    "CH3O": mol_methoxy,
    "CH2O": mol_formaldehyde,
    "CHO": mol_CHO,
}

O2 = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.208]])
O2_like = O2.copy()
O2_like.add_site_property("anchor", [True, False])
Oads_pair = [Ox, O2_like]
O2.rotate_sites(theta=np.pi / 6, axis=[1, 0, 0])
Oxo_coupling = Molecule(
    ["O", "O"],
    [
        (
            1.6815,
            0,
            0,
        ),
        (0, 0, 0),
    ],
)
Oxo_coupling.add_site_property("dimer_coupling", [True, True])

OOH_up_OH_like = OOH_up.copy()
OOH_up_OH_like.add_site_property("anchor", [True, False, False])
OH_pair = [OH, OOH_up_OH_like]
H2O = Molecule(
    ["H", "H", "O"],
    [
        [2.226191, -9.879001, 2.838300],
        [2.226191, -8.287900, 2.667037],
        [2.226191, -9.143303, 2.156037],
    ],
    site_properties={"magmom": [0.1, 0.1, 0.6], "binding_site": [False, False, True]},
)

# Deacon process
HCl = Molecule(["H", "Cl"], [[0, 0, 0], [0, 1.275, 0]])
Cl = Molecule(["Cl"], [(0, 0, 0)])
Cl2 = Molecule(["Cl", "Cl"], [[0, 0, 0], [2, 0, 0]])

# CO2 capture
CO = Molecule(["C", "O"], [[0, 0, 1.43], [0, 0, 0]])
CO.rotate_sites(theta=45, axis=[1, 0, 0])
CO2 = Molecule(
    ["C", "O", "O"],
    [
        [0, 0, 0],
        [-0.6785328, -0.6785328, -0.6785328],
        [0.6785328, 0.6785328, 0.6785328],
    ],
)
CO2_like = CO2.copy()
CO2_like.add_site_property("anchor", [False, False, True])

# Nitrate reduction
NO = Molecule("NO", [[0, 0, 1.16620], [0, 0, 0]])
NO.rotate_sites(theta=45, axis=[1, 0, 0])
NO2 = Molecule(
    ["N", "O", "O"],
    [
        [0.01706138, 4.15418327, 5.90139358],
        [-0.87449724, 4.72305561, 6.47843921],
        [0.92310187, 4.50860583, 5.19279554],
    ],
)
NO3 = Molecule(
    ["N", "O", "O", "O"],
    [
        [3.8414, 3.1696, 6.1981],
        [4.9493, 2.7581, 5.7930],
        [3.7675, 3.9390, 7.1791],
        [2.7958, 2.8052, 5.6144],
    ],
)
NO3_like = NO3.copy()
NO3_like.add_site_property("anchor", [False, False, False, True])
NO2_like = NO2.copy()
NO2_like.add_site_property("anchor", [False, False, True])

# Methanol formation
CH4 = Molecule(
    ["C", "H", "H", "H", "H"],
    [
        [2.48676400, 2.48676400, 2.48676400],
        [3.11939676, 3.11939676, 3.11939676],
        [3.11939676, 1.85413124, 1.85413124],
        [1.85413124, 1.85413124, 3.11939676],
        [1.85413124, 3.11939676, 1.85413124],
    ],
)

CH3 = CH4.copy()
CH3.rotate_sites(theta=np.pi / 4, axis=[0, 0, 1])
CH3.rotate_sites(theta=np.deg2rad(125), axis=[0, 1, 0])
CH3.remove_sites([4])
CH3.rotate_sites(theta=np.pi, axis=[0, 1, 0])

species = ["O", "C", "H", "H", "H", "H"]
coords = [
    [0.7079, 0, 0],
    [-0.7079, 0, 0],
    [-1.0732, -0.769, 0.6852],
    [-1.0731, -0.1947, -1.0113],
    [-1.0632, 0.9786, 0.3312],
    [0.9936, -0.8804, -0.298],
]
CH3OH = Molecule(species, coords)
CH3OH.rotate_sites(theta=-np.pi / 2, axis=[1, 0, 0])
CH3OH.rotate_sites(theta=np.pi / 4, axis=[0, 1, 0])

CH3OH_like = CH3OH.copy()
CH3OH_like.add_site_property("anchor", [True, False, False, False, False, False])
CH4_pair = [CH4, CH3OH_like]

# Peroxide
H2O2 = Molecule(
    ["H", "H", "O", "O"],
    [
        [1.76653038, 0.81443185, 4.81451611],
        [0.81443185, 1.76653038, 2.89052189],
        [2.28131524, 1.30263586, 4.09387932],
        [1.30263586, 2.28131524, 3.61115868],
    ],
)
H2O2.rotate_sites(theta=np.pi / 2, axis=[0, 1, 0])

Hx = Molecule(["H"], [[0, 0, 0]])
OH_like = OH.copy()
OH_like.add_site_property("anchor", [True, False])
Hads_pair = [Hx, OH_like]
Nx = Molecule(["N"], [[0, 0, 0]])
NO_like = NO.copy()
NO_like.add_site_property("anchor", [False, True])
Nads_pair = [Nx, NO_like]
Cx = Molecule(["C"], [[0, 0, 0]])
CO_like = CO.copy()
CO_like.add_site_property("anchor", [False, True])
Cads_pair = [Cx, CO_like]
adslist = [
    Oads_pair,
    Hads_pair,
    Nads_pair,
    Cads_pair,
    # monatomic adsorption of O, H, N and C can form O2,
    # OH, NO and CO with lattice positions respectively
    OH_pair,
    O2,
    [CO, CO2_like],
    H2O,
    Oxo_coupling,
]
OOH_list = [OOH_up, OOH_down]
