'''
Renderer module for gtrace.draw
'''

#{{{ Import modules

import gtrace.draw.dxf as dxf
import gtrace.draw as draw
from gtrace.unit import *
import numpy as np

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

#{{{ DXF renderer

class UnknownShapeError(BaseException):
    def __initi__(self, message):
        self.message = message

def renderDXF(canvas, filename):
    '''
    Render a canvas into a DXF file
    '''

    if canvas.unit == 'm':
        scale_factor = 1000.0
    elif canvas.unit == 'km':
        scale_factor = 1e6
    else:
        scale_factor = 1.0
        
    d = dxf.DXF(filename)

    for ly in list(canvas.layers.values()):
        d.add_layer(ly.name, color=dxf.color_encode(ly.color))
        for s in ly.shapes:
            if isinstance(s, draw.Line):
                d.add_entity(dxf.Line(np.array(s.start)*scale_factor, np.array(s.stop)*scale_factor), ly.name)
            elif isinstance(s, draw.PolyLine):
                x = np.array(s.x)*scale_factor
                y = np.array(s.y)*scale_factor
                d.add_entity(dxf.LwPolyLine(x, y), ly.name)
            elif isinstance(s, draw.Rectangle):
                d.add_entity(dxf.Rectangle(np.array(s.point)*scale_factor,
                                           s.width*scale_factor, s.height*scale_factor), ly.name)
            elif isinstance(s, draw.Circle):
                d.add_entity(dxf.Circle(np.array(s.center)*scale_factor, s.radius*scale_factor), ly.name)
            elif isinstance(s, draw.Arc):
                d.add_entity(dxf.Arc(np.array(s.center)*scale_factor, s.radius*scale_factor,
                                     rad2deg(s.startangle), rad2deg(s.stopangle)), ly.name)
            elif isinstance(s, draw.Text):
                d.add_entity(dxf.Text(s.text, np.array(s.point)*scale_factor, s.height*scale_factor, rad2deg(s.rotation)), ly.name)
            else:
                raise UnknownShapeError('Shape not supported')

    d.save_to_file()
                
#}}}
