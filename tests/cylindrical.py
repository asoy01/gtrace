#{{{ Import Modules
#First import modules

import gtrace.beam as beam
import gtrace.optcomp as opt
from gtrace.draw.tools import drawAllBeams, drawAllOptics, transAll, rotateAll
import gtrace.draw as draw
import gtrace.draw.renderer as renderer
from gtrace.unit import *
import gtrace.optics.gaussian as gauss

#Numpy is almost mandatory to use gtrace
import numpy as np
pi = np.pi

#}}}

#{{{ GaussianBeam object
#First, we create a beam object.

#Prepare a q-parameter
#Rw2q() converts ROC and the radius of a beam
#into a q-parameter. np.inf is infinity.
q0 = gauss.Rw2q(np.inf, 0.1*mm)

#Create a GaussianBeam object.
b = beam.GaussianBeam(q0=q0, wl=1064*nm)

#Set the direction angle of the beam to 10deg from the global x-axis.
b.dirAngle = deg2rad(10)

#Set the position of the origin of the beam
b.pos = (0.0, 0.0)

#}}}

#{{{ Mirror object
#Put a mirror 10cm away from the beam origin
M1 = opt.CyMirror(HRcenter=[0.1, 0], normAngleHR=pi,
                 diameter=25*cm, thickness=10*cm,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=1./1,
                 inv_ROC_AR=-1./2,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 Refl_AR=500*ppm, Trans_AR=1-500*ppm,
                 n=1.45, name='Mirror', curve_direction='h')


M1.rotate(deg2rad(-45))

#}}}

#{{{ Hit the mirror with the beam

#beams is a dictionary of the beams created by
#the injection of the beam into the mirror
beams = M1.hitFromHR(b, order=3)

#}}}

#{{{ Draw

#Create a canvas object
cnv = draw.Canvas()

#Add layers (optional)
cnv.add_layer("main_beam", color=(0,0,0))

#Draw all beams
beamList = beams.values()
drawAllBeams(cnv, beamList, drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='main_beam',
                    fontSize=0.01)

#You can also draw each beam manually. For example,
#b1 = beams['s1']
#b1.draw(cnv)
#This will draw the beam b1 into the canvas object cnv.

#Draw the mirror
M1.draw(cnv, drawName=True)

#Save the result as a DXF file
renderer.renderDXF(cnv, 'test.dxf')

#}}}

#{{{ Simple
q0 = gauss.Rw2q(np.inf, 0.1*mm)
b = beam.GaussianBeam(q0=q0, wl=1064*nm, pos=[0.0,0.0], dirAngle=0.0)

M1 = opt.Mirror(HRcenter=[2, 0], normAngleHR=deg2rad(170),
                 diameter=25*cm, thickness=10*cm,
                 wedgeAngle=deg2rad(0.25), inv_ROC_HR=1./2.3,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 n=1.45, name='Mirror')

beams = M1.hitFromHR(b, order=2)

cnv = draw.Canvas()
cnv.add_layer("main_beam", color=(0,0,0))
cnv.add_layer("main_beam_width", color=(0,0,1))
drawAllBeams(cnv, beams.values(), drawWidth=True, sigma=2.7,  layer='main_beam')
M1.draw(cnv, drawName=True)
renderer.renderDXF(cnv, 'Test.dxf')

#}}}
