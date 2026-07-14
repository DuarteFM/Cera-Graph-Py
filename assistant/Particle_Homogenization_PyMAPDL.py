# -*- coding: utf-8 -*-
"""
Created on November, 2024

@author: Thiago Assis Dutra
@email: thiagoassis.dutra@gmail.com

This code must be used only for academic research and education, i.e. no 
commercial purposes, and cannot be distributed to anyone else. When used,
this code and the author must be acknowledged in all related publications.

Important information about the input data:
- the input volumes MUST have different centers of gravity;
- the homogenized elastic constants are expressed in terms of input engineering
  constants units.

"""

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

IMPORT BLOCK

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

from ansys.mapdl.core import launch_mapdl
import os, sys
import numpy as np
import numba as nb
import time

ttot = time.time()

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

INTERNAL FUNCTIONS BLOCK

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

def CreateMaterial(myAPDL,COD,EX,EY,EZ,GXY,GXZ,GYZ,NUXY,NUXZ,NUYZ,DENS):
    """ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CREATES ORTHOTROPIC MATERIALS
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """
    # Define a material with nine engineering constants
    myAPDL.mp("EX", COD, EX)  # Elastic moduli along X
    myAPDL.mp("EY", COD, EY)  # Elastic moduli along Y
    myAPDL.mp("EZ", COD, EZ)  # Elastic moduli along Z
    myAPDL.mp("GXY", COD, GXY)  # Shear moduli on the plane XY
    myAPDL.mp("GXZ", COD, GXZ)  # Shear moduli on the plane XY
    myAPDL.mp("GYZ", COD, GYZ)  # Shear moduli on the plane XY
    myAPDL.mp("PRXY", COD, NUXY)  # Poisson's Ratio
    myAPDL.mp("PRXZ", COD, NUXZ)  # Poisson's Ratio
    myAPDL.mp("PRYZ", COD, NUYZ)  # Poisson's Ratio
    myAPDL.mp("DENS", COD, DENS)  # Density

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

def FindCenters(myAPDL,voluMat):
    """ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    FINDS THE CENTERS OF VOLUMES FROM A CERTAIN MATERIAL
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """
    voluMatCenter = []
    for i in voluMat:
        myAPDL.geometry.volume_select([i], sel_type='S', return_selected=True)
        vProps = myAPDL.vsum()
        indX = vProps.rfind('XC=')
        indY = vProps.rfind('YC=')
        indZ = vProps.rfind('ZC=')
        voluMatCenter.append([i,
                              float(vProps[indX + 4:indX + 15]),
                              float(vProps[indY + 4:indY + 15]),
                              float(vProps[indZ + 4:indZ + 15])])
    return voluMatCenter

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

@nb.jit(nopython=True)
def CompareArrays(a, b):
    """ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    DETERMINES IF TWO ARRAYS ARE IDENTITICAL
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

def StressRecover(myAPDL):
    """ * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GET RESULTS AND CREATES STIFFNESS MATRIX FOR A GIVEN LOAD CASE
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """
    #
    myAPDL.etable("", "VOLU")                   # Get element volume
    myAPDL.etable("SX", "S", "X")               # Get element stress X
    myAPDL.etable("SY", "S", "Y")               # Get element stress Y
    myAPDL.etable("SZ", "S", "Z")               # Get element stress Z
    #
    myAPDL.smult("SXV", "VOLU", "SX", 1, 1)     # Stress X by element volume
    myAPDL.smult("SYV", "VOLU", "SY", 1, 1)     # Stress Y by element volume
    myAPDL.smult("SZV", "VOLU", "SZ", 1, 1)     # Stress Z by element volume
    #
    myAPDL.ssum()
    #
    TOTVOL = myAPDL.get("val","SSUM", "", "ITEM", "VOLU")
    #
    TOTSX = myAPDL.get("val", "SSUM", "", "ITEM", "SXV")  # integrate stress
    TOTSY = myAPDL.get("val", "SSUM", "", "ITEM", "SYV")
    TOTSZ = myAPDL.get("val", "SSUM", "", "ITEM", "SZV")
    #
    C1j = TOTSX/TOTVOL
    C2j = TOTSY/TOTVOL
    C3j = TOTSZ/TOTVOL
    #
    C = np.array([C1j,C2j,C3j])
    return C

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

READ INPUT DATA

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nReading Input Data\n')

inputDataName = 'InputDataHomogenization.in'

f = open(inputDataName)

#------------------------------------------------------------------------------
#--Define the work directory
#------------------------------------------------------------------------------
f.readline()
wkdir = f.readline().replace('\n', '')
f.readline()

#------------------------------------------------------------------------------
#-- Number of the constituents of the RVE
#------------------------------------------------------------------------------
f.readline()
f.readline()
line = f.readline().replace('\n', '').split(',')

#-- Get unit cell coordinates along x direction
unitCellx0 = float(line[0])
unitCellx1 = float(line[1])

#-- Get unit cell coordinates along y direction
unitCelly0 = float(line[2]);
unitCelly1 = float(line[3]);

#-- Get unit cell coordinates along z direction
unitCellz0 = float(line[4]);
unitCellz1 = float(line[5]);

f.readline()
print('\nUnit Cell Coordinates\n')
print('x0, x1, y0, y1, z0, z1 = %.6e, %.6e, %.6e, %.6e, %.6e, %.6e \n' 
      %(unitCellx0,unitCellx1,unitCelly0,unitCelly1,unitCellz0,unitCellz1))

#------------------------------------------------------------------------------
#-- Number of the constituents of the RVE
#------------------------------------------------------------------------------
f.readline()
nConst = int(f.readline().replace('\n', ''))    #--Total number of constituents
f.readline()

#------------------------------------------------------------------------------
#-- Properties of the constituents of the RVE
#------------------------------------------------------------------------------
f.readline()
f.readline()
f.readline()
mProp = np.zeros([nConst,10])

for i in range(nConst):
    line = f.readline().replace('\n', '').split(',')
    mProp[i][:] = [float(line[0]),float(line[1]),float(line[2]),float(line[3]),
                   float(line[4]),float(line[5]),float(line[6]),float(line[7]),
                   float(line[8]),float(line[9])]
f.readline()

#------------------------------------------------------------------------------
#-- Element type
#------------------------------------------------------------------------------
f.readline()
f.readline()
try:
    eType = int(f.readline().replace('\n', ''))
except:
    print('Error in reading element type')
    sys.exit(1)
f.readline()

#------------------------------------------------------------------------------
#-- Meshing parameters for automatic (smart) element sizing
#------------------------------------------------------------------------------
f.readline()
f.readline()
f.readline()
eSmrtSz = int(f.readline().replace('\n', ''))
f.readline()

#------------------------------------------------------------------------------
#-- Element edge length on surface boundaries
#------------------------------------------------------------------------------
f.readline()
f.readline()
eSz = float(f.readline().replace('\n', ''))
f.readline()

#------------------------------------------------------------------------------
#-- Name of the .cdb file containing meshed geometry
#------------------------------------------------------------------------------
f.readline()
f.readline()
nmcdbFile = f.readline().replace('\n', '')
f.readline()

#------------------------------------------------------------------------------
#-- Name of the output result file
#------------------------------------------------------------------------------
f.readline()
f.readline()
nmOutResFile = f.readline().replace('\n', '')

f.close()

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

START MAPDL LOCALLY

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

try:
    mapdl = launch_mapdl(loglevel="WARNING", print_com=True)
except:
    print('An exception occurred!')
    sys.exit(1)
else:
    print('\nPyMAPDL connected with success!\n')
    
#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

INITIALIZATION OF PRE-PROCESSOR MODULE

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

mapdl.title('Multivolume Mesh')

#------------------------------------------------------------------------------
# Enter the model creation preprocessor
#------------------------------------------------------------------------------
mapdl.prep7()

#------------------------------------------------------------------------------
# Change the current working directory
#------------------------------------------------------------------------------
mapdl.cwd(wkdir)

#------------------------------------------------------------------------------
# Pre-define solid element types
#------------------------------------------------------------------------------
mapdl.et(1, 'SOLID185')
mapdl.et(2, 'SOLID186')
mapdl.et(3, 'SOLID187')

#------------------------------------------------------------------------------
#-- Loop through different materialss
#------------------------------------------------------------------------------
for i in range(nConst):
    #--------------------------------------------------------------------------
    #-- Create materials according to InputData.in
    #--------------------------------------------------------------------------
    CreateMaterial(mapdl,i+1,mProp[i][0],mProp[i][1],mProp[i][2],mProp[i][3],
                   mProp[i][4],mProp[i][5],mProp[i][6],mProp[i][7],mProp[i][8],
                   mProp[i][9])



#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

IMPORT THE VOLUMES

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nImporting Volumes\n')

#------------------------------------------------------------------------------
# Initializing variables
#------------------------------------------------------------------------------
voluCount = 0
voluMat = []
iMat = 1

#------------------------------------------------------------------------------
# Loop through different materials
#------------------------------------------------------------------------------
for i in range(1,nConst+1):
    #--------------------------------------------------------------------------
    # Loop through files in specified directory
    #--------------------------------------------------------------------------
    for x in os.listdir(wkdir):
        if x.endswith(str(i) + '.x_t'):
            fl_import = "~PARAIN,'" + x.split('.')[0] + "','x_t','" + wkdir +\
                        "',SOLIDS,0,0"
            mapdl.run(fl_import)
            voluCount = voluCount + 1
    #--------------------------------------------------------------------------
    # Each line of voluMat constains a range of volumes corresponding to the
    # materials, e.g. volMat[0] contains the range of volumes from Material 1 
    #--------------------------------------------------------------------------        
    voluMat.append(range(iMat, voluCount + 1))
    iMat = voluCount + 1
    
    print(str(len(voluMat[i-1])) + ' Volume(s) Type *' + str(i) + '.X_T imported\n')
    
#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

DETERMINATION OF THE CENTERS OF GRAVITY OF THE VOLUMES

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nDetermining CGs\n')

#------------------------------------------------------------------------------
# Initializing variables
#------------------------------------------------------------------------------
voluMatCenter = []

#------------------------------------------------------------------------------
# Loop through different materials
#------------------------------------------------------------------------------
for i in range(nConst):
    #--------------------------------------------------------------------------
    # Each line of voluMatCenter contains the number of the volume and its
    # respective center
    #--------------------------------------------------------------------------
    voluMatCenter.append(FindCenters(mapdl,voluMat[i]))

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

GLUE THE VOLUMES THAT ENABLES THE CREATION OF ONE SINGLE MESH

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nGluing Volumes\n')

#------------------------------------------------------------------------------
# Select all the volumes
#------------------------------------------------------------------------------
mapdl.geometry.volume_select('ALL', sel_type='S', return_selected=True)

#------------------------------------------------------------------------------
# Generate new volume by “gluing” volumes
#
# Element attributes and solid model boundary conditions assigned to the 
# original entities will not be transferred to the new entities generated
#
#------------------------------------------------------------------------------
try:
    mapdl.vglue('ALL')
except:
    mapdl.exit()
    print('Volumes could not be glued!')
    sys.exit(1)

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

DETERMINATION OF NEW CENTERS OF GRAVITY

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nDetermining new CGs\n')

#------------------------------------------------------------------------------
# Select all the volumes
#------------------------------------------------------------------------------
newVolumes = mapdl.geometry.volume_select('ALL', sel_type='S',
                                          return_selected=True)

#------------------------------------------------------------------------------
# Each line of newCenters contains the number of the volume and its
# respective center
#------------------------------------------------------------------------------
newCenters = FindCenters(mapdl,newVolumes)

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

RELATE NEW VOLUMES TO OLD VOLUMES AND ASSIGN MATERIALS

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nAssigning material to volumes\n')

print('New Volume\t|\tOld Volume\t|\tMaterial')

#------------------------------------------------------------------------------
# Loop through the new volumes
#------------------------------------------------------------------------------
for i in range(len(newVolumes)):
    a = np.array(newCenters[i][1:4])   
    #--------------------------------------------------------------------------
    # Loop through different materials
    #--------------------------------------------------------------------------
    for k in range(nConst):
        #----------------------------------------------------------------------
        # Loop through old volumes for a specific material (from Material 1)
        #----------------------------------------------------------------------
        for j in range(len(voluMat[k])):            
            b = np.array(voluMatCenter[k][j][1:4])
            #------------------------------------------------------------------
            # Check if centers of newVolume[i] and oldVolume[k] match
            #------------------------------------------------------------------
            if CompareArrays(a, b):
                print(str(newVolumes[i]) + '\t\t\t|\t' + str(voluMat[k][j]) +\
                      '\t\t\t|\t' + str(k+1))
                #--------------------------------------------------------------
                # Select the new volume
                #--------------------------------------------------------------
                mapdl.geometry.volume_select([newVolumes[i]], sel_type='S',
                                             return_selected=True)
                #--------------------------------------------------------------
                # Select assign Material k to the new volume
                #--------------------------------------------------------------
                try:
                    mapdl.vatt(k+1,'',eType,0,'')
                except:
                    print('Materials could not be assigned!')
                    sys.exit(1)
               
#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

MESH THE VOLUMES AND EXPORT MESHED GEOMETRY

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nMeshing the volumes\n')

#------------------------------------------------------------------------------
# Select all the volumes
#------------------------------------------------------------------------------
mapdl.geometry.volume_select('ALL', sel_type='S', return_selected=True)

#------------------------------------------------------------------------------
# Modify mesh parameters according to Input Data
#------------------------------------------------------------------------------
if eSmrtSz > 0:
    mapdl.smrtsize(eSmrtSz)

if eSz > 0:
    mapdl.esize(eSz)

#------------------------------------------------------------------------------
# Mesh all selected volumes
#------------------------------------------------------------------------------
if eType < 3:
    try:
        mapdl.vsweep('ALL')
    except:
        print('The volumes could not be meshed! The VSWE command is ignored')
else:
    try:
        mapdl.vmesh('ALL')
    except:
        print('The volumes could not be meshed! The VMESH command is ignored')

#%%

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

APPLY BOUNDARY CONDITIONS AND CREATE STATIC LOAD CASES

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

#------------------------------------------------------------------------------
# Solution Module
#------------------------------------------------------------------------------
mapdl.slashsolu()

#------------------------------------------------------------------------------
# Set Static Analysis
#------------------------------------------------------------------------------
mapdl.run("ANTYPE,STATIC ")

#------------------------------------------------------------------------------
# Apply Boundary Conditions related to the first column of stiffness matrix
#------------------------------------------------------------------------------
print('\nApplying Boundary Conditions related to the 1st Column of \
Homogenized Stiffness Matrix\n')

mapdl.lsclear('ALL')
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "X", unitCellx0)
mapdl.d("ALL", "UX", 0.0)
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_1
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "X", unitCellx1)
mapdl.d("ALL", "UX", unitCellx1-unitCellx0)
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Y", unitCelly0)
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "Y", unitCelly1)
mapdl.d("ALL", "UY", 0.0)
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Z", unitCellz0)
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "Z", unitCellz1)
mapdl.d("ALL", "UZ", 0.0)
#------------------------------------------------------------------------------
# Select all nodes and create load case
#------------------------------------------------------------------------------
mapdl.nsel('ALL')
mapdl.lswrite(1)

#------------------------------------------------------------------------------
# Apply Boundary Conditions related to the second column of stiffness matrix
#------------------------------------------------------------------------------
print('\nApplying Boundary Conditions related to the 2nd Column of \
Homogenized Stiffness Matrix\n')

mapdl.lsclear('ALL')
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Y", unitCelly0)
mapdl.d("ALL", "UY", 0.0)
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_1
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Y", unitCelly1)
mapdl.d("ALL", "UY", unitCelly1-unitCelly0)
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "X", unitCellx0)
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "X", unitCellx1)
mapdl.d("ALL", "UX", 0.0)
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Z", unitCellz0)
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "Z", unitCellz1)
mapdl.d("ALL", "UZ", 0.0)
#------------------------------------------------------------------------------
# Select all nodes and create load case
#------------------------------------------------------------------------------
mapdl.nsel('ALL')
mapdl.lswrite(2)

#------------------------------------------------------------------------------
# Apply Boundary Conditions related to the thir column of stiffness matrix
#------------------------------------------------------------------------------
print('\nApplying Boundary Conditions related to the 3rd Column of \
Homogenized Stiffness Matrix\n')

mapdl.lsclear('ALL')
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Z", unitCellz0)
mapdl.d("ALL", "UZ", 0.0)
#------------------------------------------------------------------------------
# Select nodes with Z coordinates = z_1
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Z", unitCellz1)
mapdl.d("ALL", "UZ", unitCellz1-unitCellz0)
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "X", unitCellx0)
#------------------------------------------------------------------------------
# Select nodes with X coordinates = x_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "X", unitCellx1)
mapdl.d("ALL", "UX", 0.0)
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_0
#------------------------------------------------------------------------------
mapdl.nsel("S", "LOC", "Y", unitCelly0)
#------------------------------------------------------------------------------
# Select nodes with Y coordinates = y_1
#------------------------------------------------------------------------------
mapdl.nsel("A", "LOC", "Y", unitCelly1)
mapdl.d("ALL", "UY", 0.0)
#------------------------------------------------------------------------------
# Select all nodes and create load case
#------------------------------------------------------------------------------
mapdl.nsel('ALL')
mapdl.lswrite(3)

#------------------------------------------------------------------------------
# Write cdb file (meshed geometry)
#------------------------------------------------------------------------------
print('\nWriting cdb file containing meshed geometry and BCs\n')
mapdl.cdwrite('DB',nmcdbFile,'cdb','','','')

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

SOLVE STATIC LOAD CASES

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nSolving system of equations\n')

mapdl.eqslv(lab='pcg')

#------------------------------------------------------------------------------
# Solve load set 1
#------------------------------------------------------------------------------
try:
    mapdl.lssolve(1, 1)
except:
    print('Load set «« Pure tension along X »» could not be solved!')

#------------------------------------------------------------------------------
# Solve load set 2
#------------------------------------------------------------------------------
try:
    mapdl.lssolve(2, 2)
except:
    print('Load set «« Pure tension along Y »» could not be solved!')

#------------------------------------------------------------------------------
# Solve load set 3
#------------------------------------------------------------------------------
try:
    mapdl.lssolve(3, 3)
except:
    print('Load set «« Pure tension along Z »» could not be solved!')

#------------------------------------------------------------------------------
# Exit solution module
#------------------------------------------------------------------------------
mapdl.finish()

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

POST-PROCESSING AND PRINTING RESULTS

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

print('\nGetting results from post-processing module\n')

#------------------------------------------------------------------------------
# Initialize Post-processor module
#------------------------------------------------------------------------------
mapdl.post1()

#------------------------------------------------------------------------------
# Solve for first column coefficients
#------------------------------------------------------------------------------
mapdl.set(1)
[C11,C21,C31] = StressRecover(mapdl)
#------------------------------------------------------------------------------
# Solve for second column coefficients
#------------------------------------------------------------------------------
mapdl.set(2)
[C12,C22,C32] = StressRecover(mapdl)
#------------------------------------------------------------------------------
# Solve for third column coefficients
#------------------------------------------------------------------------------
mapdl.set(3)
[C13,C23,C33] = StressRecover(mapdl)
#------------------------------------------------------------------------------
#-- Stiffness Matrix for normal load cases
#------------------------------------------------------------------------------
C1 = np.array([[C11,C12,C13],
               [C21,C22,C23],
               [C31,C32,C33]])
#------------------------------------------------------------------------------
#-- Compliance Matrix for normal load cases
#------------------------------------------------------------------------------
S1 = np.linalg.inv(C1)

#------------------------------------------------------------------------------
#-- Print results
#------------------------------------------------------------------------------
print('\n*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *')
print('\nHomogenized Elastic Constants\n')
print('E1, E2, E3 = %.6e, %.6e, %.6e \n' %(1./S1[0,0],1./S1[1,1],1./S1[2,2]))
print('v12, v13, v23 = %0.4f, %0.4f, %0.4f\n' %(-S1[1,0]/S1[0,0],
                                           -S1[2,0]/S1[1,1],
                                           -S1[2,1]/S1[1,1]))
print('\n*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *')

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

SAVING RESULTS INTO OUTPUT FILE

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

fid = open(nmOutResFile + '.out', 'w')
fid.write('Input File Path:\n')
fid.write(wkdir + '\n')

fid.write('\nUnit Cell Coordinates\n')
fid.write('      x0           x1\n')
fid.write('%.6e %.6e\n' %(unitCellx0,unitCellx1))
fid.write('      y0           y1\n')
fid.write('%.6e %.6e\n' %(unitCelly0,unitCelly1))
fid.write('      z0           z1\n')
fid.write('%.6e %.6e\n' %(unitCellz0,unitCellz1))

for i in range(0,nConst):
    fid.write('\n Material Number = %d - Mechanical Properties\n' %(i+1))
    fid.write('E1, E2, E3 = %.6E, %.6E, %.6E\n' %(mProp[i][0],mProp[i][1],
                                                  mProp[i][2]))
    fid.write('G12, G13, G23 = %.6E, %.6E, %.6E\n' %(mProp[i][3],mProp[i][4],
                                                        mProp[i][5]))
    fid.write('v12, v13, v23 = %.5f, %.5f, %.5f\n' %(mProp[i][6],mProp[i][7],
                                                     mProp[i][8]))
    fid.write('rho = %.6E\n' %(mProp[i][9]))

fid.write('\nHomogenized Elastic Constants\n')
fid.write('E1, E2, E3 = %.6e, %.6e, %.6e \n' %(1./S1[0,0],1./S1[1,1],1./S1[2,2]))
fid.write('v12, v13, v23 = %0.4f, %0.4f, %0.4f\n' %(-S1[1,0]/S1[0,0],
                                                    -S1[2,0]/S1[1,1],
                                                    -S1[2,1]/S1[1,1]))
fid.write('\n*  *  *  *  *  *  *  *  *  *  *')
fid.write('\nHomogenized Stiffness Matrix - Normal Load Cases')
fid.write('\n ')
fid.write('% .5E % .5E % .5E\n' %(C1[0,0],C1[0,1],C1[0,2]))
fid.write('% .5E % .5E % .5E\n' %(C1[1,0],C1[1,1],C1[1,2]))
fid.write('% .5E % .5E % .5E\n' %(C1[2,0],C1[2,1],C1[2,2]))
fid.write('\n*  *  *  *  *  *  *  *  *  *  *')

fid.close()

#%%
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

CLOSE CURRENT SESSION

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

#------------------------------------------------------------------------------
#-- Close PyMapdl session
#------------------------------------------------------------------------------
mapdl.exit()
ttot = time.time() - ttot

print ('Total elapsed time = %10.1f min' %(ttot/60.))

