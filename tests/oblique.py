#{{{ ==== Import modules ====

#Add the local "modules" directory to the python's module search path.
#This trick only works if you run this script from the root directory of OptLayout.
import sys
import os
sys.path.append(os.path.abspath(os.curdir))

from matplotlib.font_manager import FontProperties, FontManager
import gtrace.beam as beam
import gtrace.optcomp as opt
from gtrace.nonsequential import non_seq_trace
from gtrace.draw.tools import drawAllBeams, drawAllOptics, transAll, rotateAll, drawOptSys
import gtrace.draw as draw
import gtrace.draw.renderer as renderer
from gtrace.unit import *
import gtrace.optics.gaussian as gauss
from gtrace.optics.cavity import Cavity
import gtrace.optics.geometric as geom
import numpy as np
pi = np.pi
import scipy as sp
import scipy.optimize as sopt
import copy
from scipy.constants import c
from multiprocessing import Process, Pipe, cpu_count
import time
import sharedmem as shm
from paralleltools import distribute1DArray, parallelize
import pickle

#}}}

#{{{ Parameters

MC_Dia = 10.0*cm
MC_Thick = 3.0*cm
MCe_ROC = 40.0
MMT1_ROC = -24.6
MMT2_ROC = 24.3

MCi_Refl = 0.9937
MCo_Refl = 0.9937
MCe_Refl = 0.9999
AR_Refl = 0.1/100

pos_MCi = np.array([-27.759072, 0.136953797]) 
pos_MCo = np.array([-27.259072, 0.136953797]) 
pos_MCe = np.array([-27.509048, 26.53565317]) 
pos_MMT1 = np.array([-21.394048, -0.060390856]) 
pos_MMT2 = np.array([-24.479048, 0.290123756]) 
LMMT = np.linalg.norm(pos_MMT1 - pos_MMT2)
L_MC_MMT1 = 5.8357910693793

Lmc = (np.linalg.norm(pos_MCi - pos_MCo) + np.linalg.norm(pos_MCi - pos_MCe) + np.linalg.norm(pos_MCo - pos_MCe))/2

MC = Cavity(r1=0.9, r2=0.9, L=Lmc, R1=-1e8, R2=MCe_ROC, power=True)

MCq0 = MC.waist()[0]

inputBeamDict = {}



nsilica = 1.44967
nsilica_green = 1.46071
nsaph = 1.754
nsaph_green = 1.7717



#}}}

#{{{ Define mirrors


MCi = opt.Mirror(HRcenter=[0,0], normAngleHR=0.0,
                 diameter=MC_Dia, thickness=MC_Thick,
                 wedgeAngle=-deg2rad(2.5), inv_ROC_HR=0.0,
                 Refl_HR=MCi_Refl, Trans_HR=1-MCi_Refl,
                 Refl_AR=AR_Refl, Trans_AR=1-AR_Refl,
                 n=nsilica, name='MCi')

MCo = opt.Mirror(HRcenter=[0,0], normAngleHR=0.0,
                 diameter=MC_Dia, thickness=MC_Thick,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=0.0,
                 Refl_HR=MCo_Refl, Trans_HR=1-MCo_Refl,
                 Refl_AR=AR_Refl, Trans_AR=1-AR_Refl,
                 n=nsilica, name='MCo')

MCe = opt.Mirror(HRcenter=[0,0], normAngleHR=0.0,
                 diameter=MC_Dia, thickness=MC_Thick,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=1.0/MCe_ROC,
                 Refl_HR=MCe_Refl, Trans_HR=1-MCe_Refl,
                 Refl_AR=AR_Refl, Trans_AR=1-AR_Refl,
                 n=nsilica, name='MCe')

MMT1 = opt.Mirror(HRcenter=[0,0], normAngleHR=0.0,
                 diameter=MC_Dia, thickness=MC_Thick,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=1.0/MMT1_ROC,
                 Refl_HR=0.999, Trans_HR=1-0.999,
                 Refl_AR=AR_Refl, Trans_AR=1-AR_Refl,
                 n=nsilica, name='MMT1')

MMT2 = opt.Mirror(HRcenter=[0,0], normAngleHR=0.0,
                 diameter=MC_Dia, thickness=MC_Thick,
                 wedgeAngle=deg2rad(2.5), inv_ROC_HR=1.0/MMT2_ROC,
                 Refl_HR=0.999, Trans_HR=1-0.999,
                 Refl_AR=AR_Refl, Trans_AR=1-AR_Refl,
                 n=nsilica, name='MMT2')


inputOptics = [MCi, MCo, MCe, MMT1, MMT2]

#}}}

#{{{ Put MC mirrors

MCi.HRcenter = pos_MCi
MCo.HRcenter = pos_MCo
MCe.HRcenter = pos_MCe

v1 = pos_MCi - pos_MCo
v2 = pos_MCe - pos_MCo
v1 = v1/np.linalg.norm(v1)
v2 = v2/np.linalg.norm(v2)
MCo.normVectHR = (v1+v2)/2

v2 = pos_MCe - pos_MCi
v2 = v2/np.linalg.norm(v2)
MCi.normVectHR = (-v1+v2)/2

MCe.normVectHR = np.array([0.0, -1.0])

#}}}

#{{{ Propagate MC beams

bmc = beam.GaussianBeam(q0 = MCq0, pos=(pos_MCi + pos_MCo)/2, dirAngle=0.0)

beams = MCo.hitFromHR(bmc, order=1)

b1=beams['input'].copy()
b1.propagate(b1.length)
b2 = beams['t1'].copy()

L=np.real(b2.qy - b1.qy)*nsilica

theta1=0.780663677706396
theta2=0.5068847403834057
theta3=0.46325150908354695
theta4=0.70469103773435282

c1=np.cos(theta1)
c2=np.cos(theta2)
c3=np.cos(theta3)
c4=np.cos(theta4)

a=c4*c2/(c3*c1)
b=c4*c1*L/(c3*c2*nsilica)
c=0
d=c1*c3/(c2*c4)
q=b1.qx

(a*q+b)/(c*q+d)

#}}}
