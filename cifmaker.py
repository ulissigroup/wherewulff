import json
from os.path import join
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


fdir = '/global/cscratch1/sd/ibenlo/mongo_results/block_2021-12-19-20-42-33-847568/launcher_2022-01-11-04-11-54-197117/'
fname = 'TiCr(RuO4)2_NM_stability_analys.json'

with open(join(fdir,fname), 'r') as f:
    data = json.load(f)

struct = Structure.from_dict(data['structure'])

CifWriter(struct).write_file(join(fdir, 'slab.cif'))
