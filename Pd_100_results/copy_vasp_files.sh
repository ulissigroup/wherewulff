#!/bin/bash

# Current dict and full path
work_dir="${PWD##*/}"
CWD=$(pwd)

# VASP INPUTS
vasp_inps="{INCAR,POSCAR,POTCAR,KPOINTS}"

# VASP OUTPUTS
vasp_outs="{OUTCAR,OSZICAR,CONTCAR,vasprun.xml,vasp.out}"

# Either inputs or outputs
if [ $work_dir = 'inputs' ]; then
    scp $1:$2/$vasp_inps $CWD
else
    scp $1:$2/$vasp_outs $CWD
fi

echo "Done!"


