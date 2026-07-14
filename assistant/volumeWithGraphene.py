# -*- coding: utf-8 -*-
"""
Created on May, 2024

@author: Duarte Félix Macedo
@email: duarte.felix.macedo@ubi.pt

This code must be used only for academic research and education, i.e. no 
commercial purposes, and cannot be distributed to anyone else. When used,
this code and the author must be acknowledged in all related publications.

Important information about the input data:
- the input RVE from neper software and graphene properties
- create volumes with graphene

"""

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

IMPORT BLOCK

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

import os
import math
import random
import gmsh
import numpy as np

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

INITIALIZATION AND READ INPUT DATA

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

# File name
newfile = 'nfrom_morpho-id7'


# Initialize Gmsh API 
gmsh.initialize()

gmsh.open("..\\" + newfile + ".geo")
arq = open("..\\" + newfile + '.stcell', 'r')

# Read type of volume

text = arq.readlines()
newParticle = []
phaseParticle = []

for linha in text:
    palavras = linha.split()
    newParticle.append(int(palavras[1]))
    phaseParticle.append(int(palavras[0]))

arq.close()

# Get RVE size

L = max([max(gmsh.model.getValue(0, x[1], []))
         for x in gmsh.model.getEntities(0)])


gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates() 
gmsh.model.geo.synchronize()


# Material Properties

densityGraphene = 2.350367 # Graphene density (g/cm^3)

fractionMassGraphene = 0.015 # mass fraction

fractionGraphenePostSintering = 50 # mass fraction after sintering (%)

density = 2.5572 # ceramic composite density (g/cm^3)

volGrapheneMax = L*L*L*fractionMassGraphene*fractionGraphenePostSintering/100
volGrapheneMax = volGrapheneMax*density/densityGraphene # Graphene volume

volumeGraphene = 0

space = 0
nn =[0,0,1]

grapheneSheetThickness = 0.00034 # thickness of graphene sheets

numberSheets = 71 # number of graphene sheets

thickness = grapheneSheetThickness*numberSheets # thickness of graphene

areaMaxGrafeno = volGrapheneMax/thickness # total graphene area


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

GET VOLUMES DATA

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """


ListVolume = [x[1] for x in gmsh.model.getEntitiesInBoundingBox(0-space,
                                                                 0-space,
                                                                 0-space,
                                                                 L+space,
                                                                 L+space,
                                                                 L+space, 
                                                                 3)]


maxPartivleNumber = max(ListVolume)


gmsh.model.getEntitiesInBoundingBox(0-space, 
                                    0-space, 
                                    0-space, 
                                    L+space, 
                                    L+space, 
                                    L+space, 
                                    2)
gmsh.model.getEntitiesInBoundingBox(0+space, 
                                    0+space, 
                                    0+space, 
                                    L-space, 
                                    L-space, 
                                    L-space, 
                                    2)



ListSurfVolume = [[y[1] for y in gmsh.model.getBoundary([(3, x)])] for x in ListVolume]



PointsSurf = [[[[[b[1] for b in a]   
                   for a in [gmsh.model.getBoundary([(1, z[1])],False)]]  
                  for z in gmsh.model.getBoundary([(2, y)])]  for y in x] 
                for x in ListSurfVolume]


sizeMaxVolumes = []

for i in range(len(PointsSurf)):
	maxi = [0,0,0]
	mini = [L,L,L]
	for j in range(len(PointsSurf[i])):
		for l in range(len(PointsSurf[i][j])):
			for m in range(len(PointsSurf[i][j][l])):

				a0 = gmsh.model.getValue(0, PointsSurf[i][j][l][m][0], [])
				maxi = [max([maxi[0],a0[0]]),max([maxi[1],a0[1]]),max([maxi[2],a0[2]])]
				mini = [min([mini[0],a0[0]]),min([mini[1],a0[1]]),min([mini[2],a0[2]])]
	bo = maxi[0] - mini[0]
	co = maxi[1] - mini[1]
	do = maxi[2] - mini[2]
	sizeMaxVolumes.append([ListVolume[i],min(bo,co,do)])



volumesToKill = [x[0] for x in sizeMaxVolumes if x[1] <= thickness]


for x in volumesToKill: 

	ListSurfVolume.pop(ListVolume.index(x))
	PointsSurf.pop(ListVolume.index(x))
	ListVolume.remove(x)



centroid = [[ListVolume[i] , [[[0,0,0] , [0,0,0] , 0] 
                              for _ in range(len(PointsSurf[i]))]] 
            for i in range(len(PointsSurf))]


for i in range(len(PointsSurf)):
	for j in range(len(PointsSurf[i])):
		for l in range(len(PointsSurf[i][j])):
			for m in range(len(PointsSurf[i][j][l])):
				centroid[i][1][j][0] += gmsh.model.getValue(0, PointsSurf[i][j][l][m][0], [])
		centroid[i][1][j][0] = [n/len(PointsSurf[i][j]) for n in centroid[i][1][j][0]]
		centroid[i][1][j][1] = [-ListSurfVolume[i][j]/abs(ListSurfVolume[i][j])*n 
                          for n in gmsh.model.getNormal(abs(ListSurfVolume[i][j]), 
                                                        gmsh.model.getParametrization(2,
                                                                                      abs(ListSurfVolume[i][j]),
                                                                                      centroid[i][1][j][0]))]
		centroid[i][1][j][2] = abs(ListSurfVolume[i][j])
		


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

DEFINE GRAPHENE POSITION

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

ListRamdom = [x[1] for x in gmsh.model.getEntities(2)]

ListRamdomNummmmmm = [x[1] for x in gmsh.model.getEntities(2)]

 

random.shuffle(ListRamdom)
 
NumFGraf = np.ones(len(ListRamdom))
NumFGraf = NumFGraf*numberSheets


areaGrapheneTotal = 0

SurfToScrape = []
SurfToScrapeRamdom = []

for ij in range(len(ListRamdom)):

	x = ListRamdom[ij]

	PointsFaceAux = [[[c[1] for c in b]   
                   for b in [gmsh.model.getBoundary([(1, a[1])],False)]]  
                  for a in gmsh.model.getBoundary([(2, x)])]
	PointsFaceAux = [a[0][0] for a in PointsFaceAux]

	center = [0,0,0]
	for y in PointsFaceAux:
		center += gmsh.model.getValue(0, y, [])

	center = [y/len(PointsFaceAux) for y in center]
	
	if 0 not in center and L not in center:
		

		sAux = gmsh.model.geo.copy([(2, x)])

		normal = gmsh.model.getNormal(x, gmsh.model.getParametrization(2,x,center))
        
        

		normalRotation = [nn[1]*normal[2]-nn[2]*normal[1],
                    nn[2]*normal[0]-nn[0]*normal[2],
                    nn[0]*normal[1]-nn[1]*normal[0]]

		angle = math.acos(nn[0]*normal[0]+nn[1]*normal[1]+nn[2]*normal[2])
		angle = angle/math.sqrt(nn[0]*nn[0]+nn[1]*nn[1]+nn[2]*nn[2])
		angle = angle/math.sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);

		gmsh.model.geo.translate(sAux, -center[0], -center[1], -center[2])

		if [0,0,0] != normalRotation: gmsh.model.geo.rotate(sAux, 0, 0, 0,
                                                      -normalRotation[0],
                                                      -normalRotation[1],
                                                      -normalRotation[2], 
                                                      angle)

	
		gmsh.model.geo.synchronize()


		PointsFaceAux = [[[c[1] for c in b]   
                    for b in [gmsh.model.getBoundary([(1, a[1])],False)]]  
                   for a in gmsh.model.getBoundary(sAux)]
		PointsFaceAux = [a[0][0] for a in PointsFaceAux]


		auxiliar = 0

		for i in range(len(PointsFaceAux)-1):

			p0 = gmsh.model.getValue(0, PointsFaceAux[i], [])
			p1 = gmsh.model.getValue(0, PointsFaceAux[i+1], [])


			auxiliar += p0[0]*p1[1] - p0[1]*p1[0];

		p0 = gmsh.model.getValue(0, PointsFaceAux[-1], [])
		p1 = gmsh.model.getValue(0, PointsFaceAux[0], [])




		auxiliar += p0[0]*p1[1] - p0[1]*p1[0];

		areaGrapheneNow = abs(auxiliar/2);
        
		areaGrapheneTotal += areaGrapheneNow;
        
		volumeGraphene += areaGrapheneNow*grapheneSheetThickness*NumFGraf[ij]
        
 
		gmsh.model.geo.remove(sAux,True)

		SurfToScrape.append(x)
		SurfToScrapeRamdom.append(NumFGraf[ij])



	if volumeGraphene >= volGrapheneMax: break



""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

CREATE NEW VOLUME OF CERAMICS AND POROS

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """


gmsh.model.geo.synchronize()


gmsh.model.removeEntities(gmsh.model.getEntities(3))
gmsh.model.removeEntities(gmsh.model.getEntities(2))
gmsh.model.removeEntities(gmsh.model.getEntities(1))
gmsh.model.removeEntities(gmsh.model.getEntities(0))


newVolumes = []

for x in centroid:

	xx = x[0]
	gmsh.model.occ.addBox(0, 0, 0, L, L, L, xx)
    
	for y in x[1]:

		if y[1][0]*y[1][1] != 0 or y[1][1]*y[1][2] != 0 or y[1][2]*y[1][0] != 0:

			if y[2] not in SurfToScrape:
				VolumeAux = gmsh.model.occ.addBox(-2*L,-2*L,-4*L, 4*L, 4*L, 4*L)
			else:
				VolumeAux = gmsh.model.occ.addBox(-2*L,
                                      -2*L,
                                      -4*L+SurfToScrapeRamdom[SurfToScrape.index(y[2])]*grapheneSheetThickness/2,
                                      4*L,
                                      4*L,
                                      4*L)



			normalRotation = [nn[1]*y[1][2]-nn[2]*y[1][1],
                     nn[2]*y[1][0]-nn[0]*y[1][2],
                     nn[0]*y[1][1]-nn[1]*y[1][0]]

			angle = math.acos(nn[0]*y[1][0]+nn[1]*y[1][1]+nn[2]*y[1][2])
			angle = angle/math.sqrt(nn[0]*nn[0]+nn[1]*nn[1]+nn[2]*nn[2])
			angle = angle/math.sqrt(y[1][0]*y[1][0]+y[1][1]*y[1][1]+y[1][2]*y[1][2]);


			if [0,0,0] != normalRotation: gmsh.model.occ.rotate([(3,VolumeAux)],
                                                       0,
                                                       0,
                                                       0,
                                                       normalRotation[0],
                                                       normalRotation[1],
                                                       normalRotation[2],
                                                       angle)


			gmsh.model.occ.translate([(3,VolumeAux)], y[0][0], y[0][1], y[0][2])	


			VolumeAux = gmsh.model.occ.cut([(3, xx)], [(3, VolumeAux)])

			xx = VolumeAux[0][0][1]


	gmsh.model.occ.synchronize()
    
	newVolumes.append("particulas STEP/particula_"+
                   str(ListVolume[centroid.index(x)])+
                   "_"+
                   str(phaseParticle[newParticle.index(ListVolume[centroid.index(x)])])+".step")
	
	gmsh.write("particulas STEP/particula_"+str(ListVolume[centroid.index(x)])+
            "_"+
            str(phaseParticle[newParticle.index(ListVolume[centroid.index(x)])])+".step")	

	gmsh.model.occ.remove([(3,ListVolume[centroid.index(x)])],True)


gmsh.model.removeEntities(gmsh.model.getEntities(3))
gmsh.model.removeEntities(gmsh.model.getEntities(2))
gmsh.model.removeEntities(gmsh.model.getEntities(1))
gmsh.model.removeEntities(gmsh.model.getEntities(0))


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

CREATE GRAPHENE VOLUMES

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """

gmsh.model.occ.synchronize()


for i in range(len(newVolumes)):
	if i == 0:
		gmsh.open(newVolumes[i])
	else:
		gmsh.merge(newVolumes[i])
     
gmsh.model.occ.synchronize()

ListVolumesFinal = [x[1] for x in gmsh.model.getEntities(3)]


volumeGraphene = [(3,max(ListVolumesFinal)+1)]


gmsh.model.occ.addBox(0, 0, 0, L, L, L, volumeGraphene[0][1])


volumeGrapheneAux = []


for x in ListVolumesFinal:
    
	volumeGrapheneAux.append((3, x))

volumeGraphene = gmsh.model.occ.cut(volumeGraphene,volumeGrapheneAux)


gmsh.model.occ.synchronize()


ListVolumesFinal = [x[1] for x in gmsh.model.getEntities(3)]

gmsh.write("particulas STEP/particula_grafeno_3.step")	

gmsh.model.removeEntities(gmsh.model.getEntities(3))
gmsh.model.removeEntities(gmsh.model.getEntities(2))
gmsh.model.removeEntities(gmsh.model.getEntities(1))
gmsh.model.removeEntities(gmsh.model.getEntities(0))



gmsh.open("particulas STEP/particula_grafeno_3.step")


ListVolume = gmsh.model.getEntities(3)


gmsh.model.removeEntities(gmsh.model.getEntities(3))
gmsh.model.removeEntities(gmsh.model.getEntities(2))
gmsh.model.removeEntities(gmsh.model.getEntities(1))
gmsh.model.removeEntities(gmsh.model.getEntities(0))

for x in ListVolume:

	gmsh.open("particulas STEP/particula_grafeno_3.step")
    
    
    
	pointsReais = []
    

	surfaces_ = gmsh.model.getBoundary([x])
	areas_ = gmsh.model.getBoundary(surfaces_,False)
	points_ = gmsh.model.getBoundary(areas_,False)

	for y in gmsh.model.getEntities(3):
		if y != x:
			gmsh.model.removeEntities([y])


	for y in gmsh.model.getEntities(2):
		if y not in surfaces_:
			gmsh.model.removeEntities([y])



	for y in gmsh.model.getEntities(1):
		if y not in areas_:
			gmsh.model.removeEntities([y])



	for y in gmsh.model.getEntities(0):
		if y not in points_:
			gmsh.model.removeEntities([y])
		else:
			pointsReais.append(y)
            
                
	xyzMax = [0,0,0]
	xyzMin = [L,L,L]
    
	for y in pointsReais:
		xxyyzz = gmsh.model.getValue(0, y[1], [])  
		xyzMax[0] = max([xxyyzz[0],xyzMax[0]]) 
		xyzMax[1] = max([xxyyzz[1],xyzMax[1]]) 
		xyzMax[2] = max([xxyyzz[2],xyzMax[2]]) 
		xyzMin[0] = min([xxyyzz[0],xyzMin[0]]) 
		xyzMin[1] = min([xxyyzz[1],xyzMin[1]]) 
		xyzMin[2] = min([xxyyzz[2],xyzMin[2]])
        
	

    

	if thickness/2 < max(xyzMax[0]-xyzMin[0],xyzMax[1]-xyzMin[1],xyzMax[2]-xyzMin[2]):
		gmsh.write("particulas STEP/particula_"+str(x[1]+maxPartivleNumber)+"_3.step")	



	gmsh.model.removeEntities(gmsh.model.getEntities(3))
	gmsh.model.removeEntities(gmsh.model.getEntities(2))
	gmsh.model.removeEntities(gmsh.model.getEntities(1))
	gmsh.model.removeEntities(gmsh.model.getEntities(0))



""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

ElIMINATE PARTICLES TOO SMALL

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """





for i in range(len(newVolumes)):
   
	gmsh.open(newVolumes[i])
    
    
	ListVolume = gmsh.model.getEntities(3)
    
	pointsReais = []
    
    

	surfaces_ = gmsh.model.getBoundary(ListVolume)
	areas_ = gmsh.model.getBoundary(surfaces_,False)
	points_ = gmsh.model.getBoundary(areas_,False)

	for y in gmsh.model.getEntities(3):
		if y != x:
			gmsh.model.removeEntities([y])


	for y in gmsh.model.getEntities(2):
		if y not in surfaces_:
			gmsh.model.removeEntities([y])



	for y in gmsh.model.getEntities(1):
		if y not in areas_:
			gmsh.model.removeEntities([y])



	for y in gmsh.model.getEntities(0):
		if y not in points_:
			gmsh.model.removeEntities([y])
		else:
			pointsReais.append(y)
            
                
	xyzMax = [0,0,0]
	xyzMin = [L,L,L]
    
	for y in pointsReais:
		xxyyzz = gmsh.model.getValue(0, y[1], [])  
		xyzMax[0] = max([xxyyzz[0],xyzMax[0]]) 
		xyzMax[1] = max([xxyyzz[1],xyzMax[1]]) 
		xyzMax[2] = max([xxyyzz[2],xyzMax[2]]) 
		xyzMin[0] = min([xxyyzz[0],xyzMin[0]]) 
		xyzMin[1] = min([xxyyzz[1],xyzMin[1]]) 
		xyzMin[2] = min([xxyyzz[2],xyzMin[2]])
        
	

    

	if thickness > min(xyzMax[0]-xyzMin[0],xyzMax[1]-xyzMin[1],xyzMax[2]-xyzMin[2]):
		os.remove(newVolumes[i])	



	gmsh.model.removeEntities(gmsh.model.getEntities(3))
	gmsh.model.removeEntities(gmsh.model.getEntities(2))
	gmsh.model.removeEntities(gmsh.model.getEntities(1))
	gmsh.model.removeEntities(gmsh.model.getEntities(0))

""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

FINISH PROGRAM

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * """


os.remove("particulas STEP/particula_grafeno_3.step")




gmsh.finalize()
