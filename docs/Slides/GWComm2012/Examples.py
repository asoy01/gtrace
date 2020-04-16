'''
Generate example outputs
'''

#{{{ Import Modules
#First import modules

import gtrace.beam as beam
import gtrace.optcomp as opt
from gtrace.draw.tools import drawAllBeams, drawAllOptics, transAll, rotateAll
import gtrace.draw as draw
import gtrace.draw.renderer as renderer
from gtrace.unit import *
import gtrace.optics.gaussian as gauss
from gtrace.nonsequential import non_seq_trace

#Numpy is almost mandatory to use gtrace
import numpy as np
pi = np.pi

#}}}

#{{{ GaussianBeam object

#q-parameter of the beam
q0 = gauss.Rw2q(ROC=np.inf, w=0.3*mm)

#Create a GaussianBeam object.
b1 = beam.GaussianBeam(q0=q0, wl=1064*nm, length=30*cm, P=1.0)

#Set the direction angle of the beam to 10deg from the global x-axis.
b1.dirAngle = deg2rad(10)

#Set the position of the origin of the beam
b1.pos = (0.0, 0.0)

#}}}

#{{{ Mirror object

M1 = opt.Mirror(HRcenter=[50*cm, 10*cm], normAngleHR=pi,
                 diameter=25*cm, thickness=10*cm,
                 wedgeAngle=deg2rad(0.25), inv_ROC_HR=1./(120*cm),
                 inv_ROC_AR=0,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 Refl_AR=500*ppm, Trans_AR=1-500*ppm,
                 n=1.45, name='M1')

M2 = opt.Mirror(HRcenter=[0*cm, 18*cm], normAngleHR=deg2rad(5.0),
                 diameter=15*cm, thickness=5*cm,
                 wedgeAngle=deg2rad(0.25), inv_ROC_HR=-1./(350*cm),
                 inv_ROC_AR=0,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 Refl_AR=500*ppm, Trans_AR=1-500*ppm,
                 n=1.45, name='M2')

M3 = opt.Mirror(HRcenter=[30*cm, 30*cm], normAngleHR=deg2rad(21.3),
                 diameter=15*cm, thickness=5*cm,
                 wedgeAngle=deg2rad(1), inv_ROC_HR=1./(350*cm),
                 inv_ROC_AR=0,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 Refl_AR=500*ppm, Trans_AR=1-500*ppm,
                 n=1.45, name='M3')


#}}}

#{{{ Sequential

#{{{ Hit the mirror with the beam

#beams is a dictionary of the beams created by
#the injection of the beam into the mirror
beams1 = M1.hitFromHR(b1, order=2)
b2 = beams1.pop('r1')
b2.length=20*cm
beams2 = M2.hitFromHR(b2, order=2)
b3 = beams2.pop('r1')
b3.length=20*cm
beams3 = M3.hitFromAR(b3, order=2)

#}}}

#{{{ Draw

#Create a canvas object
cnv = draw.Canvas()

#Draw all beams

drawAllBeams(cnv, beams1.values(), drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='main_beam1',
                    fontSize=0.01)

drawAllBeams(cnv, beams2.values(), drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='main_beam2',
                    fontSize=0.01)

drawAllBeams(cnv, beams3.values(), drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='main_beam3',
                    fontSize=0.01)

drawAllBeams(cnv, [b1,b2,b3], drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='source_beam',
                    fontSize=0.01)

#Draw the mirror
drawAllOptics(cnv, [M1,M2,M3])

#Save the result as a DXF file
renderer.renderDXF(cnv, 'SeqTrace.dxf')

#}}}

#}}}

#{{{ Non-Sequential

#{{{ Hit the mirrors

beams = non_seq_trace([M1,M2,M3], b1, order=30, power_threshold=1e-6)

#}}}

#{{{ Draw

#Create a canvas object
cnv = draw.Canvas()

#Draw all beams

drawAllBeams(cnv, beams, drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='main_beam',
                    fontSize=0.01)

drawAllBeams(cnv, [b1], drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='source_beam',
                    fontSize=0.01)

#Draw the mirror
drawAllOptics(cnv, [M1,M2,M3])

#Save the result as a DXF file
renderer.renderDXF(cnv, 'NonSeq.dxf')

#}}}

#}}}

#{{{ Misc

#{{{ A Gaussian Beam

#q-parameter of the beam
q0 = gauss.Rw2q(ROC=np.inf, w=0.03*mm)

#Create a GaussianBeam object.
b1 = beam.GaussianBeam(q0=q0, wl=1064*nm, length=6*cm, P=1.0)

b1.propagate(-3*cm)

#Set the direction angle of the beam to 10deg from the global x-axis.
b1.dirAngle = deg2rad(0)

#Set the position of the origin of the beam
b1.pos = (0.0, 0.0)

#Create a canvas object
cnv = draw.Canvas()

#Draw all beams

drawAllBeams(cnv, [b1], drawWidth=True, sigma=3.0, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer='source_beam',
                    fontSize=0.01)


#Save the result as a DXF file
renderer.renderDXF(cnv, 'GaussBeam.dxf')

#}}}

#}}}


