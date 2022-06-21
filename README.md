<h1 align="center">WhereWulff <br/> (Under Development)</h1>


[![WhereWulff](https://github.com/ulissigroup/mo-wulff-workflow/actions/workflows/WhereWulff.yml/badge.svg)](https://github.com/ulissigroup/mo-wulff-workflow/actions/workflows/WhereWulff.yml)

## Introduction

`WhereWulff` couples deep expertise in Quantum Chemistry and Catalysis with that in workflow engineering, an approach that is slowly gaining traction in the science community.

<figure align="center">
	<img src="img/wherewulff_img.png">
	<figcaption>
		<p><b>Figure 1.</b> WhereWulff general schema that consists in the bulk workflow to get the
		equilibrium bulk structure with the most stable magnetic configuration as NM, AFM or FM (Left), and the reactivity workflow that analyzes Wulff Construction, Surface Pourbaix diagram and OER Reactivity for a given material (Right).
		</p>
	</figcaption>
</figure>

## Installation

After installing [conda](http://conda.pydata.org/), run the following commands to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named wherewulff and install dependencies.

```bash
conda env create -f wherewulff_env.yml
conda activate wherewulff
pip install -e .
```

`WhereWulff` main dependencies are [FireWorks](https://materialsproject.github.io/fireworks/), [Atomate](https://atomate.org) and [Pymatgen](https://pymatgen.org), that need further installation steps.

### FireWorks and Atomate

We refer the user to the [Atomate](https://atomate.org/installation.html) installation documentation to have a deeper explanation on how to set-up `FireWorks/Atomate` properly.

### Pymatgen

[Pymatgen](https://pymatgen.org) needs the `.pmgrc.yml` file to be configured with the VASP pseudopotentials, default DFT functional and the [Materials Project]() API token as:

To configure Pymatgen to find the VASP pseudopotential see [POTCAR setup](https://pymatgen.org/installation.html#)

```bash
pmg config -p <EXTRACTED_VASP_POTCAR> <MY_PSP>
pmg config --add PMG_VASP_PSP_DIR <MY_PSP>
pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54
```

Is always good practice to test if Pymatgen is able to find a given POTCAR file. The following command should create a new POTCAR file for H atom:

```bash
pmg potcar -s H -f PBE_54
```

Don't forget to include your `PMG_MAPI_KEY` to been able to run the Stability Analysis at the end of the Bulk Workflow.

Your .pmgrc.yml file should look like:
```bash
PMG_DEFAULT_FUNCTIONAL: PBE_54
PMG_MAPI_KEY: "YOUR_API_TOKEN"
PMG_VASP_PSP_DIR: "POTCAR_DIR"
```

## Run the Workflow

The following example is how to load the Bulk Workflow to the launchpad and then submitting how to submit it through the FireWorks command line:

```python
from CatFlows.launchers.bulkflows import BulkFlows

# Import CIF file
cif_file = "<<YOUR_CIF_FILE>>"

# BulkFlow method and config
bulk_flow = BulkFlows(bulk_structure=cif_file,
                      n_deformations=21,
					  nm_magmom_buffer=0.6,
					  conventional_standard=True)

# Get Launchpad
launchpad = bulk_flow.submit(
    hostname="localhost",
    db_name="<<DB-NAME>>",
    port="<<DB-PORT>>",
    username="<<DB-USERNAME>>",
    password="<<DB-PASSWORD>>",
)
```

The Bulk workflow is called through the BulkFlow method which is able to submit the workflow to the launchpad for a given `CIF` file consisting in a bulk structure of a metal or metal oxide material.

The user needs to provide the `CIF` file pathway and the configure the workflow in terms of number of deformations for the `EOS` (Equation of States), the magnetic buffer for non-magnetic species included in the given material and whether to transform the given structure to conventional standrad.

The submit method inside BulkFlows class needs the MongoDB configuration features such as `hostname`, `db_name`, `port`, `username` and `password`. We encourage the user to not make public this information.

We encourage the user to use `Fireworks webgui` to make sure the workflow is properly added to the launchpad. Finally the way to run the workflow through the command line shell is as follows (-m flag is for maximum 5 jobs running in parallel): 

```bash
qlaunch rapidfire -m 5
```

The surface chemistry workflow is called through the CatFlows method which is able to submit the whole worklfow to the launchpad for a given `CIF` file consisting in a bulk structure.

```python
from CatFlows.launchers.catflows import CatFlows

# Import CIF file
cif_file = "<<YOUR_CIF_FILE>>"

# CatFlows method and config
cat_flows = CatFlows(cif_file, exclude_hkl=[(1, 0, 0), (1, 1, 1), (0, 0, 1)])

# Get Launchpad
launchpad = cat_flows.submit(
    hostname="localhost",
    db_name="<<DB-NAME>>",
    port="<<DB-PORT>>",
    username="<<DB-USERNAME>>",
    password="<<DB-PASSWORD>>",
)
```

The user needs to provide a `CIF` file pathway, preferably as a result of running the bulk workflow beforehand so then the bulk structure will be with the equilibrium cell parameters and with the magnetic configuration well defined. CatFlows can be extensibly configured depending to the user needs see [documentation](https://github.com/ulissigroup/mo-wulff-workflow/blob/main/CatFlows/launchers/catflows.py). The submit function inside CatFlows works in the same way as BulkFlows by providing the required information to being able to connect to the MongoDB database and the launchpad.

Finally, submitting the workflow must be done through the same command as the previous examples:

```bash
qlaunch rapidfire -m 5
```

## Acknowledgements

We acknowledge NRC, CMU, UofT and NERSC?

## License

`WhereWulff` is released under the [MIT](https://github.com/ulissigroup/mo-wulff-workflow/blob/main/LICENSE.md)

## Citing `WhereWulff`
```bibtex
@article{wherewulff2022,
	title = {WhereWulff: An Autonomous Workflow to Democratize and Scale Complex Material Discovery for Electocatalysis},
	author = {Rohan Yuri Sanspeur, Javier Heras-Domingo and Zachary Ulissi},
	journal = {To be submitted},
	year = {2022},
}
```
