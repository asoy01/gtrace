#{{{ Imports

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.draw as draw
import gtrace.draw.renderer as renderer

#}}}

#{{{ Author and License Infomation

#Copyright (c) 2011-2012, Yoichi Aso
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__author__ = "Yoichi Aso"
__copyright__ = "Copyright 2011-2012, Yoichi Aso"
__credits__ = ["Yoichi Aso"]
__license__ = "BSD"
__version__ = "0.2.0"
__maintainer__ = "Yoichi Aso"
__email__ = "aso@granite.phys.s.u-tokyo.ac.jp"
__status__ = "Beta"

#}}}

#{{{ Draw optical system

def drawOptSys(optList, beamList, filename, fontSize=False):

    d = draw.Canvas()
    d.unit = 'm'

    d.add_layer("main_beam", color=(255,0,0))
    d.add_layer("main_beam_width", color=(255,0,255))    
    d.add_layer("stray_beam", color=(0,255,0))
    d.add_layer("stray_beam_width", color=(0,255,255))    

    for b in beamList:
        if b.stray_order > 0:
            b.layer = 'stray_beam'
            sigma = 1.0
            drawWidth=False
        else:
            b.layer = 'main_beam'
            sigma = 3.0
            drawWidth=True
            
        b.draw(d, sigma=sigma, drawWidth=drawWidth, drawPower=True, drawName=True, fontSize=fontSize)
    
    drawAllOptics(d, optList, drawName=True)

    render.renderDXF(d, filename)
    
#}}}

#{{{ Draw all beams

def drawAllBeams(d, beamList, sigma=3.0, drawWidth=True, drawPower=False,
                 drawROC=False, drawGouy=False, drawOptDist=False, layer=None, mode='x',
                    fontSize=0.01):
    
    for ii in range(len(beamList)):
        if layer is not None:
            beamList[ii].layer = layer
            
        beamList[ii].draw(d, sigma=sigma, mode=mode, drawWidth=drawWidth, drawPower=drawPower,
                    drawROC=drawROC, drawGouy=drawGouy, drawOptDist=drawOptDist,
                    fontSize=fontSize)

#}}}

#{{{ Draw all optics
def drawAllOptics(d, opticsList, drawName=True, layer=None):
    for ii in range(len(opticsList)):
        if layer is not None:
            opticsList[ii].layer = layer

        opticsList[ii].draw(d, drawName=drawName)

#}}}

#{{{ Translate all

def transAll(objList, transVect):
    for ii in range(len(objList)):
        objList[ii].translate(transVect)
#}}}

#{{{ Rotata all

def rotateAll(objList, angle, center):
    for ii in range(len(objList)):
        objList[ii].rotate(angle, center)
#}}}
