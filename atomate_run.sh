#!/usr/bin/env bash

#atomate_run.sh for workflows in python scripts.

#Variables
workflow=main_bulk_local.py

#Add structure and workflow
mongod &>/dev/null & python $workflow
