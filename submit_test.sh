#!/usr/bin/env bash

#atomate_run.sh for workflows in python scripts.

#Variables
workflow=main_slab_test.py

#Add structure and workflow
python $workflow

# Lpad test
lpad get_fws
