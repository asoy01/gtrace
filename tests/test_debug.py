import gtrace.beam as beam
import gtrace.optcomp as opt
from gtrace.draw.tools import drawAllBeams, drawAllOptics, transAll, rotateAll
import gtrace.draw as draw
import gtrace.draw.renderer as renderer
from gtrace.unit import *
import gtrace.optics.gaussian as gauss



#Create a GaussianBeam object.
q0 = gauss.Rw2q(np.inf, 0.1*mm)
b = beam.GaussianBeam(q0=q0, wl=1064*nm)

b.dirAngle = deg2rad(10)
b.pos = (0.0, 0.0)

M1 = opt.Mirror(HRcenter=[0.1, 0], normAngleHR=pi,
                 diameter=25*cm, thickness=10*cm,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=1./10,
                 inv_ROC_AR=0,
                 Refl_HR=0.9, Trans_HR=1-0.9,
                 Refl_AR=500*ppm, Trans_AR=1-500*ppm,
                 n=1.45, name='Mirror')