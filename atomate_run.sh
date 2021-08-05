#!/usr/bin/env bash

#atomate_run.sh for workflows in python scripts.

#Variables
workflow=main.py

#Add structure and workflow
mongod &>/dev/null & python $workflow
