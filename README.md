# Computational modeling of polycrystalline ceramic-graphene composites: RVE generation and Young's modulus prediction via numerical simulation.

## Workflow Overview

This repository contains a computational workflow for generating polycrystalline ceramic RVEs with graphene inclusions and predicting their effective Young's modulus via finite element simulation.

## Step 1: Generate Polycrystalline RVE (without graphene)

The base polycrystalline RVE is generated using Neper (https://neper.info) with a bimodal grain size distribution. Example command:
```
neper -T -n from\_morpho -domain "cube(L,L,L)" -o nfrom\_morpho-id7 -id n \\
  -morpho "diameq:fr1\*normal(m1,d1)+fr2\*normal(m2,d2),1-sphericity:lognormal(0.145,0.03)" \\
  -format obj,tess,ori,geo -statcell "mode,id,diameq,size,sphericity"
```
#### Parameter description:

- L — Edge length of the cubic RVE

- n — Unique ID controlling the random seed for microstructure generation

- fr1 — Volume fraction of phase 1

- fr2 = (1 - fr1) — Volume fraction of phase 2

- m1, d1 — Mean and variance of equivalent diameter for phase 1

- m2, d2 — Mean and variance of equivalent diameter for phase 2

Note: Output files (\*.tess, \*.geo, \*.obj, \*.ori) should be saved in the main repository directory.

## Step 2: Introduce Graphene Inclusions

Navigate to the assistant/ directory and run the python script "volumeWithGraphene.py". This script inserts graphene platelets into the RVE and exports the individual volumes (grains, pores, and graphene) as .STEP files.

## Step 3: Convert STEP to Parasolid Format

Using the SolidWorks macro "convertFileVolumes.swp", convert the .STEP files to Parasolid (.X\_T) format. Prerequisite: SolidWorks installation with macro execution enabled.

## Step 4: Finite Element Homogenization

Run the python script "Particle\_Homogenization\_PyMAPDL.py". This script:

* Imports the .X\_T geometry files
* Assembles and combines the sub-volumes into a single RVE
* Computes the effective Young's modulus of the composite (RVE geometry and material properties are defined in the "InputDatagenization.in" file).

## Dependencies

* Neper
* Python
* SolidWorks
* ANSYS MAPDL

## Notes

* Ensure all file paths in the python script match your local directory structure.
* The assistant/ directory must exist and contain the required input files before executing Steps 2–4.





