'''
Drawing classes for gtrace
'''

#{{{ Import modules
import numpy as np
pi = np.pi
#}}}

#{{{ Author and License Infomation

#Copyright (c) 2011-2021, Yoichi Aso
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
__copyright__ = "Copyright 2011-2021, Yoichi Aso"
__credits__ = ["Yoichi Aso"]
__license__ = "BSD"
__version__ = "0.2.4"
__maintainer__ = "Yoichi Aso"
__email__ = "yoichi.aso@nao.ac.jp"
__status__ = "Beta"

#}}}

#{{{ Canvas class

class Canvas(object):
    '''
    Canvas class
    '''

    def __init__(self, unit='m'):
        self.layers = {}
        self.unit = unit

    def add_layer(self, name, color=(0,0,0)):
        self.layers[name] = Layer(name, color=color)

    def add_shape(self, shape, layername):
        if not layername in self.layers:
            self.add_layer(layername)

        self.layers[layername].add_shape(shape)

#}}}

#{{{ Layer class

class Layer(object):
    '''
    Layer class
    '''

    def __init__(self, name, color=(0,0,0)):
        self.name = name
        self.color = color
        self.shapes = []

    def add_shape(self, shape):
        self.shapes.append(shape)
        
#}}}

#{{{ Shape class

class Shape(object):
    '''
    Shape class
    '''

    def __init__(self):
        pass
        
#}}}

#{{{ Line class

class Line(Shape):
    '''
    Line class
    '''

    def __init__(self, start, stop, thickness=0):
        super(Line, self).__init__()
        self.start = start
        self.stop = stop
        self.thickness = thickness
        
#}}}

#{{{ PolyLine

class NumberOfElementError(BaseException):
    def __initi__(self, message):
        self.message = message
        
class PolyLine(Shape):
    '''
    A light weight poly-line
    '''
    
    def __init__(self, x, y, thickness=0):
        '''
        = Arguments =
        x: x coordinates of the vertices
        y: y coordinates of the vertices        
        '''
        super(PolyLine, self).__init__()
        self.x = x
        self.y = y
        if len(x) != len(y):
            raise NumberOfElementError('The numbers of elements of x and y do not match.')
        self.numpoints = len(x)
        self.thickness = thickness


#}}}

#{{{ Rectangle

class Rectangle(Shape):
    '''
    A rectangle
    '''
    
    def __init__(self, point, width, height, thickness=0):
        '''
        = Arguments =
        point: lower left corner of the rectangle
        width:
        height:
        '''
        super(Rectangle, self).__init__()
        self.point = point
        self.width = width
        self.height = height
        self.thickness = thickness


#}}}

#{{{ Circle

class Circle(Shape):
    '''
    A circle
    '''

    def __init__(self, center, radius, thickness=0):
        super(Circle, self).__init__()
        self.center = center
        self.radius = radius
        self.thickness = thickness

#}}}

#{{{ Arc

class Arc(Shape):
    '''
    An arc

    Note that angles are stored in rad.
    '''

    def __init__(self, center, radius, startangle, stopangle, thickness=0, angle_in_rad=True):
        super(Arc, self).__init__()
        self.center = center
        self.radius = radius
        self.thickness = thickness
        if angle_in_rad:
            self.startangle = startangle
            self.stopangle = stopangle
        else:
            self.startangle = pi*startangle/180.0
            self.stopangle = pi*stopangle/180.0
            

#}}}

#{{{ Text

class Text(Shape):
    '''
    Text

    Note that angles are stored in rad.    
    '''

    def __init__(self, text, point, height=1.0, rotation=0.0, angle_in_rad=True):
        super(Text, self).__init__()
        self.text = text
        self.point = point
        self.height = height
        if angle_in_rad:        
            self.rotation = rotation
        else:
            self.rotation = pi*rotation/180.0

#}}}
