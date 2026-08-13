'''
Drawing classes for gtrace
'''

#{{{ Import modules
import numpy as np
pi = np.pi
#}}}

#{{{ Author and License Infomation

#Copyright (c) 2011-2026, Yoichi Aso
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
__copyright__ = "Copyright 2011-2026, Yoichi Aso"
__credits__ = ["Yoichi Aso"]
__license__ = "BSD"
__version__ = "0.5.0"
__maintainer__ = "Yoichi Aso"
__email__ = "asoy01@gmail.com"
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

class NumberOfElementError(Exception):
    '''
    Raised when a polyline is given x and y of different lengths.

    Derived from Exception, not BaseException: see UnknownShapeError in
    renderer.py for why that matters. The constructor was likewise spelt
    __initi__, so the message was dropped.
    '''
    pass


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
    A rectangle, square to the axes or turned about a point of its own.

    Note that angles are stored in rad.

    The rectangle is built from ``point``, ``width`` and ``height``
    with its sides along the axes, and then turned by ``angle`` about
    ``pivot``. Those four numbers are what it is; the corners follow
    from them, and :meth:`corners` is where every part of gtrace that
    draws, bounds or picks one gets them from.

    The pivot is kept rather than folded into the corner, so that a
    rectangle turned about a hole, a hinge or the middle of a bench
    still knows what it was turned about: setting ``angle`` again
    turns it about the same point. It is carried when the rectangle
    is moved, since a turn about a point that stayed behind is a turn
    about a different part of the shape.
    '''

    def __init__(self, point, width, height, thickness=0, angle=0.0,
                 pivot=None, angle_in_rad=True):
        '''
        = Arguments =
        point: lower left corner of the rectangle, before it is turned
        width:
        height:
        thickness:
        angle: how far it is turned, counterclockwise
        pivot: the point it is turned about, in the same coordinates as
        point. None - the default - is the middle of the rectangle, so
        that a rectangle given an angle and nothing else turns where it
        stands.
        angle_in_rad: whether angle is in radians. False is degrees.
        '''
        super(Rectangle, self).__init__()
        self.point = point
        self.width = width
        self.height = height
        self.thickness = thickness
        self.angle = float(angle) if angle_in_rad else pi*float(angle)/180.0
        self.pivot = (None if pivot is None
                      else [float(pivot[0]), float(pivot[1])])

    def pivot_point(self):
        '''
        The point the rectangle is turned about, as a point.

        The middle of the rectangle where none was given - which is
        why a rectangle that is moved and never given a pivot keeps
        turning about itself.
        '''
        if self.pivot is not None:
            return np.asarray(self.pivot, dtype='float64')
        p = np.asarray(self.point, dtype='float64')
        return p + [self.width/2.0, self.height/2.0]

    def turned(self, angle, pivot=(0.0, 0.0)):
        '''
        A copy turned about a point, still a rectangle.

        The turn is added to the one the rectangle already carries,
        and what it is turned about follows: a rectangle that turned
        about a hole in it turns about where that hole has got to. A
        rectangle that had no pivot of its own is left with none, so
        it goes on turning about its own middle - which is where its
        middle has got to, since the middle travels with the corners.

        = Arguments =
        angle: how far to turn it, counterclockwise, in radians
        pivot: the point to turn about, in the same coordinates as
        point. Default the origin.

        Returns a new Rectangle.
        '''
        pv = np.asarray(pivot, dtype='float64')
        ca = np.cos(angle)
        sa = np.sin(angle)

        def carry(p):
            d = np.asarray(p, dtype='float64') - pv
            return pv + [d[0]*ca - d[1]*sa, d[0]*sa + d[1]*ca]

        # The corners are turned by adding the angle and moving what
        # the rectangle is turned about; the box it is written from
        # follows that point rather than being turned itself, which is
        # what keeps the two statements of the same rectangle in step.
        q = self.pivot_point()
        moved = carry(q) - q
        return Rectangle(np.asarray(self.point, dtype='float64') + moved,
                         self.width, self.height, thickness=self.thickness,
                         angle=self.angle + angle,
                         pivot=None if self.pivot is None else carry(q))

    def corners(self):
        '''
        The four corners, lower left first and counterclockwise, as
        they stand after the turn.

        Returns
        -------
        numpy.ndarray
            Of shape (4, 2).
        '''
        p = np.asarray(self.point, dtype='float64')
        cs = np.array([p,
                       p + [self.width, 0.0],
                       p + [self.width, self.height],
                       p + [0.0, self.height]])
        if self.angle == 0.0:
            return cs
        q = self.pivot_point()
        ca = np.cos(self.angle)
        sa = np.sin(self.angle)
        R = np.array([[ca, -sa], [sa, ca]])
        return (cs - q) @ R.T + q


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
