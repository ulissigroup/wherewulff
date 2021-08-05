# MO Wulff Workflow

![workflow](img/mo_wulff_workflow.png)

A starting point for a general Metal oxide Surfaces Workflow able to build the Wulff construction from DFT calculations automatized by Pymatgen, FireWorks and Atomate.

# Goals

The goal of this repository is to make accessible the code to everyone in group and
improve it in a collaborative way. Hopefully, ending on a great tool for many applications
in our group and more.

# Javi ToDo List

- [x] Metal oxide surfaces as non-dipolar, symmetric and Stoichiometric
- [x] Optimize both oriented bulk + slab model (VASP)
- [x] Surface Energy task after optimization
- [x] Wulff shape Analysis (as separeted Task)
- [ ] Decorate bulk structure with magnetic moments (MAGMOM) based on crystal field theory
- [ ] Add Unittests with **fake_vasp**
- [ ] Benchmark RuO2 and IrO2

# Richard ToDo List

- [x] Basic rotational DoF for OH and OOH (not generalized for other 85 molecules yet).
- [ ] Metal oxide surfaces with dipolar moment (oc21-dataset)
- [ ] Non-stoichiometric slab models (Terminations) + gamma_hkl (with chemical potential)
- [ ] Include GGA+U method
- [ ] Single point calculation with hybrid DFT Functional
- [ ] **IDEA** Include defects in slabs

# Yuri ToDo List

- [ ] Generate unique hashes to identify/query jobs in database
- [ ] Look into passing job info from parent to child fireworks
- [ ] Add ContinueOptimizeFW to refactor branch
- [ ] Add a script that runs the jobs in the launchpad across computing resources

# Yuri/Javi ToDo List

- [ ] Clean-up code for ADS_SLAB generation and add to refactor
