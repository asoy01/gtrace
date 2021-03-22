'''
gtrace.beam

A module to define GaussianBeam class.

'''

#{{{ Import modules

import numpy as np
pi = np.pi
array = np.array
sqrt = np.lib.scimath.sqrt
from numpy.linalg import norm

from traits.api import HasTraits, Int, Float, CFloat, CComplex, CArray, List, Str

from .unit import *
import gtrace.optics as optics
import gtrace.optics.geometric
from gtrace.optics.gaussian import q2zr, q2w, q2R, optimalMatching
import copy
#import gtrace.sdxf as sdxf
import gtrace.draw as draw
import scipy.optimize as sopt

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
__copyright__ = "Copyright 2011-2021, Yoichi Aso"
__credits__ = ["Yoichi Aso"]
__license__ = "BSD"
__version__ = "0.2.1"
__maintainer__ = "Yoichi Aso"
__email__ = "yoichi.aso@nao.ac.jp"
__status__ = "Beta"

#}}}

#{{{ GaussianBeam Class

class GaussianBeam(HasTraits):
    '''
        
    This is a class to represent a Gaussian beam.
    A GaussianBeam object has its origin (pos) and a propagation
    direction (dirVect or dirAngle).
    A GaussianBeam is characterized by q-parameter(s) at its origin.
    The beam can be either circular or elliptic. In order to deal with
    elliptic beams, some parameters are stored in pairs like (q0x, q0y).
    x and y denote the axes of the cross section of the beam. x-axis is
    parallel to the paper and the y-axis is perpendicular to the paper.

    A beam object can be propagated through a free space or made
    to interact with an optics.

    As a beam propagate through optical system, optical distance and
    Gouy phase are accumerated.

    *=== Attributes ===*
    
    **q:** q-parameter of the beam. If the beam is eliptic, q is the q-parameter
        of the best matching circular mode.
        
    **qx:** q-parameter of the beam in the x-direction. [complex float]
    
    **qy:** q-parameter of the beam in the y-direction. [complex float]
    
    **pos:** Position of the beam origin. [(2,) float array]
    
    **dirVect:** Propagation direction vector. [(2,) float array]
    
    **dirAngle:** Propagation direction angle measured from the positive x-axis. [float]
              
    **length:** Length of the beam (used for DXF export) [float]
    
    **layer:** Layer name of the beam when exported to a DXF file.

    **name:** Name of the beam
              
    **wl:** Wavelength in vacuum. Not the wavelength in the medium. [float]

    **n:** Index of refraction of the medium the beam is passing through. [float]

    **P:** Power [float]
    
    **wx:** Beamwidth in x-direction [float]

    **wy:** Beamwidth in y-direction [float]

    **optDist:** Accumulated optical distance.

    **Gouyx:** Accumulated Gouy phase in x-direction.

    **Gouyy:** Accumulated Gouy phase in y-direction.

    **Mx:** ABCD matrix in x-direction.
           This is a 2x2 matrix representing the product
           of ABCD transformations applied to this beam.
           It defaults to an identity matrix.
           Whenever a beam experience an ABCD matrix
           transformation, such as propagation in the space
           or reflection by a curved mirror, the applied ABCD
           matrix is multiplied to this matrix, so that we can
           keep track of what kind of transformations were
           made during beam propagation.

    **My:** ABCD matrix in y-direction. The meaning is the same as Mx.

    **departSurfAngle:** The angle formed by x-axis and the normal vector of
                                  the surface from which the beam is departing.
                                  Default is None. Used by the drawing routine.

    **departSurfInvROC:** Inverse of the ROC of the surface from which the beam is departing.
                                     The ROC is positive for a concave surface seen from
                                     the beam side.
                                     Default is None. Used by the drawing routine.                                    
    
    **incSurfAngle:** The angle formed by the x-arm and the normal vector of
                            the surface to which the beam is incident.
                            Default is None. Used by the drawing routine.    

    **incSurfInvROC:** Inverse of the ROC of the surface to which the beam is incident.
                               The ROC is positive for a concave surface seen from
                               the beam side.
                               Default is None. Used by the drawing routine.

    **stray_order:** An integer indicating if this beam is a stray light or not.
                        The default value is 0. Every time a beam is reflected by an AR surface
                        or transmits an HR surface, this couter is increased by 1.

    '''
#{{{ Traits Definitions

    name = Str()
    wl = CFloat(1064.0*nm)  #Wavelength
    P = CFloat(1*W)  #Power
    q = CComplex()  #q-parameter at the origin (best matching circular mode)
    qx = CComplex(1j)  #q-parameter at the origin (x-direction)
    qy = CComplex(1j)  #q-parameter at the origin (y-direction)
    qrx = CComplex()  #Reduced q-parameter at the origin (x-direction)
                                  # qrx = qx/n
    qry = CComplex()  #Reduced q-parameter at the origin (x-direction)
                                  # qry = qy/n
    Gouyx = CFloat(0.0) #Accumurated Gouy phase
    Gouyy = CFloat(0.0) #Accumurated Gouy phase
    wx = CFloat()
    wy = CFloat()
    n = CFloat(1.0)

    pos = CArray(dtype=np.float64, shape=(2,))
    length = CFloat(1.0)
    layer = Str()
    dirVect = CArray(dtype=np.float64, shape=(2,))
    dirAngle = CFloat()
    optDist = CFloat(0.0)

    Mx = CArray(value=[[1,0],[0,1]],dtype=np.float64, shape=(2,2))
    My = CArray(value=[[1,0],[0,1]],dtype=np.float64, shape=(2,2))
    
#}}}

#{{{ __init__

    def __init__(self, q0=1j*2*pi/(1064*nm)*1e-6/2, q0x=False, q0y=False,
                 pos=[0.0,0.0], length=1.0, dirAngle=0.0, dirVect=None,
                 wl=1064*nm, P=1*W, n=1.0, name="Beam", layer='main_beam'):
        '''
        === Arguments ===

        q0: q-parameter. If q0x or q0y is not given, this parameter is used
            as both qx and qy. [complex float]
            
            q0x: q-parameter in x-direction [complex float]
            q0y: q-parameter in y-direction [complex float]
            
            pos: Position of the origin of the beam [(2,) float array]
            length: Length of the beam (used for DXF export) [float]
            dirAngle: Propagation direction angle measured from the positive x-axis.
                      [float]    
                      dirVect: Propagation direction vector. [(2,) float array]
                      wl: wavelength [float]
                      P: Power [float]
                      n: Index of refraction of the medium the beam is passing through. [float]
                      name: Name of the beam
                      layer: Layer name of the beam when exported to a DXF file.
        '''
        self.wl = wl
        self.P = P
        self.pos = pos
        self.length = length
        self.name = name
        self.layer = layer
        self.n = n
        
        if q0x:
            self.qx = q0x
        else:
            self.qx = q0
            
        if q0y:
            self.qy = q0y
        else:
            self.qy = q0

        if dirVect != None:
            self.dirVect = dirVect
        else:
            self.dirAngle = dirAngle
            self._dirAngle_changed(0,0)

        self.optDist = 0.0

        self.departSurfAngle = None
        self.departSurfInvROC = None        
        self.incSurfAngle = None
        self.incSurfInvROC = None
        self.stray_order = 0

#}}}

#{{{ copy

    def copy(self):
        '''
        Make a deep copy.
        '''
        b = copy.deepcopy(self)
        b.qrx = self.qrx
        b.qry = self.qry
        return b

#}}}

#{{{ propagate

    def propagate(self, d):
        '''
        Propagate the beam by a distance d from the current position.
        self.n is used as the index of refraction.
        During this process, the optical distance traveled is added
        to self.optDist.
        self.Goux and self.Gouyy are also updated to record the Gouy
        phase change.
        '''
        qx0 = self.qx
        qy0 = self.qy

        ABCD = np.array([[1.0, d/self.n],[0.0, 1.0]])
        #ABCD = np.array([[1.0, d/self.n**2],[0.0, 1.0]])
        self.ABCDTrans(ABCD)
        self.pos = self.pos + self.dirVect*d

        #Increase the optical distance
        self.optDist = self.optDist + self.n*d

        #Increase the Gouy phase
        self.Gouyx = self.Gouyx + np.arctan(np.real(self.qx)/np.imag(self.qx))\
                     - np.arctan(np.real(qx0)/np.imag(qx0))

        self.Gouyy = self.Gouyy + np.arctan(np.real(self.qy)/np.imag(self.qy))\
                     - np.arctan(np.real(qy0)/np.imag(qy0))

#}}}

#{{{ ABCD Trans
    def ABCDTrans(self, ABCDx, ABCDy=None):
        '''
        Apply ABCD transformation to the beam.

        **ABCDx:** ABCD matrix for x-direction

        **ABCDy:** ABCD matrix for y-direction
        '''
        if ABCDy is None:
            ABCDy = ABCDx
            
        #Update q-parameters
        self.qrx = (ABCDx[0,0]*self.qrx + ABCDx[0,1])/(ABCDx[1,0]*self.qrx + ABCDx[1,1])
        self.qry = (ABCDy[0,0]*self.qry + ABCDy[0,1])/(ABCDy[1,0]*self.qry + ABCDy[1,1])

        #Update Mx and My
        self.Mx = np.dot(ABCDx, self.Mx)
        self.My = np.dot(ABCDy, self.My)        
        
#}}}

#{{{ rotate

    def rotate(self, angle, center=False):
        '''
        Rotate the beam around 'center'.
        If center is not given, the beam is rotated
        around self.pos.

        *Arguments*

        **angle:** Rotation angle in radians.

        **center:** Center for rotation. [(2,) array of float]
        '''        
        if center:
            center = np.array(center)
            pointer = self.pos - center
            pointer = optics.geometric.vector_rotation_2D(pointer, angle)
            self.pos = center + pointer
            
        self.dirAngle = self.dirAngle + angle

#}}}

#{{{ Translate

    def translate(self, trVect):
        '''
        Translate the beam by the direction and the distance
        specified by a vector.

       *Arguments*

        **trVect:** A vector to specify the translation direction and
                distance. [(2,) float array]
        '''        
        trVect = np.array(trVect)
        self.pos = self.pos + trVect
        
#}}}

#{{{ Flip
    def flip(self, flipDirVect=True):
        '''
        Change the propagation direction of the beam
        by 180 degrees.
        This is equivalent to the reflection of the beam
        by a spherical mirror with the same ROC as the beam.

        If optional argument flipDirVect is set to False,
        the propagation direction of the beam is not changed.
        '''
        self.qx = -np.real(self.qx)+1j*np.imag(self.qx)
        self.qy = -np.real(self.qy)+1j*np.imag(self.qy)
        if flipDirVect:
            self.dirVect = - self.dirVect

#}}}

#{{{ width

    def width(self, dist):
        '''
        Returns the beam width at a distance dist
        from the origin of the beam.
        The width is the radius where the light power becomes 1/e^2.
        '''
        dist = np.array(dist)
        k = 2*pi/(self.wl/self.n)
        qx = self.qx + dist
        qy = self.qy + dist

        return (np.sqrt(-2.0/(k*np.imag(1.0/qx))), np.sqrt(-2.0/(k*np.imag(1.0/qy))))

#}}}

#{{{ R

    def R(self, dist=0.0):
        '''
        Returns the beam ROC at a distance dist
        from the origin of the beam.
        '''
        dist = np.array(dist)
        k = 2*pi/self.wl
        qx = self.qx + dist/self.n
        qy = self.qy + dist/self.n

        return (q2R(qx), q2R(qy))

#}}}

#{{{ Waist

    def waist(self):
        '''
        Return the tuples of waist size and distance
        '''

        #Waist positions
        dx = -np.real(self.qx)
        dy = -np.real(self.qy)

        #Waist sizes
        wx = self.width(dx)[0]
        wy = self.width(dy)[1]

        return {"Waist Size":(wx, wy), "Waist Position":(dx, dy)}
    
#}}}

#{{{ draw

#{{{ Main Function

    def draw(self, cv, sigma=3., mode='x', drawWidth=True,
             fontSize=False, drawPower=False, drawROC=False, drawGouy=False,
             drawOptDist=False, drawName=False, debug=False):
        '''
        Draw the beam into a DXF object.

        *Arguments*

        **cv:** draw.Canvas object.
        
        **sigma:** The width of the beam drawn is sigma * (1/e^2 radius of the beam).
               The default is sigma = 3. sigma = 2.7 gives 1ppm diffraction loss.
               [float]
               
        **mode:** 'avg', 'x', or 'y'. A beam can have different widths for x- and y-
              directions. If 'avg' is specified, the average of them are drawn.
              'x' and 'y' specifies to show the width of the respective directions.
        
        **fontSize:** Size of the font used to show supplemental informations. [float]

        **drawWidth:** Whether to draw width or not. [Boolean]

        **drawPower:** Whether to show the beam power. [Boolean]

        **drawROC:** Whether to show the ROC or not. [Boolean]

        **drawGouy:** Whether to show the Gouy phase or not. [Boolean]

        **drawOptDist:** Whether to show the accumulated optical distance or not. [Boolean]

        **drawName:** Whether draw the name of the beam or not. [Boolean]
        '''
        if not fontSize:
            fontSize = self.width(self.length/2)[0]*sigma/5

        start = tuple(self.pos)
        stop = tuple(self.pos + self.dirVect * self.length)

        #Location to put texts
        #mid = self.pos + self.dirVect * self.length/2
        #side = mid+fontSize*1.2*k        
        k = np.array((-self.dirVect[1], self.dirVect[0]))
        text_location = tuple(self.pos + (self.length - 10*fontSize)*self.dirVect + k*fontSize*1)

        
        #Draw the center line
        cv.add_shape(draw.Line(start, stop), self.layer)

        if drawWidth:
            self.drawWidth(cv, sigma, mode)
            
        annotation = ''
        if drawName:
            annotation = annotation+'%s '%self.name
            
        if drawPower:
            annotation = annotation+'P=%.2E '%self.P

        if drawROC:
            annotation = annotation+'ROCx=%.2E '%q2R(self.qx) 
            annotation = annotation+'ROCy=%.2E '%q2R(self.qy)             

        if drawGouy:
            annotation = annotation+'Gouyx=%.2E '%self.Gouyx
            annotation = annotation+'Gouyy=%.2E '%self.Gouyy

        if drawOptDist:
            annotation = annotation+'Optical distance=%.2E '%self.optDist

        cv.add_shape(draw.Text(text=annotation, point=text_location, 
                             height=fontSize), layername='text')

        #Indicate the beam direction
        # text_location = tuple(self.pos + 10*fontSize*self.dirVect +k*fontSize*1)
        # dxf.append(sdxf.Text(text=self.name+':origin P=%.2E'%self.P, point=text_location,
        #                      height=fontSize, layer='text'))

        # text_location = tuple(self.pos + (self.length - 10*fontSize)*self.dirVect + k*fontSize*1)
        # dxf.append(sdxf.Text(text=self.name+':end P=%.2E'%self.P, point=text_location,
        #                      height=fontSize, layer='text'))
        
#}}}

#{{{ drawWidth

    def drawWidth(self, cv, sigma, mode):
        #Draw the width

        #Determine the number of line segments to draw
        zr = q2zr(self.qx)
        resolution = zr/10.0
        if resolution > self.length/10.0:
            resolution = self.length/10.0

        numSegments = int(self.length/resolution)

        if numSegments > 100:
            numSegments = 100

        #q0 to be used
        if mode == 'x':
            q0 = self.qx
        elif mode == 'y':
            q0 = self.qy
        else:
            q0 = (self.qx + self.qy)/2.0

        #Determine the start z coordinates
        if self.departSurfAngle is not None:
            theta = self.departSurfAngle - self.dirAngle
            k = 2*pi/(self.wl/self.n)
            if np.abs(self.departSurfInvROC) > 1.0/100:
                R = 1.0/self.departSurfInvROC
                (z1u, z1d) = optimStartPointR(theta, R, q0, k, sigma)
            else:
                (z1u, z1d) = optimCrossPointFlat(theta, q0, k, sigma)
        else:
            z1u = 0.0
            z1d = 0.0

        #Determine the end z coordinates
        if self.incSurfAngle is not None:
            theta = np.mod(self.incSurfAngle+pi, 2*pi) - self.dirAngle
            k = 2*pi/(self.wl/self.n)            
            if np.abs(self.incSurfInvROC) > 1.0/100:
                R = 1.0/self.incSurfInvROC
                (z2u, z2d) = optimEndPointR(theta, R, q0+self.length, k, sigma)
            else:
                (z2u, z2d) = optimCrossPointFlat(theta, q0+self.length, k, sigma)
        else:
            z2u = 0.0
            z2d = 0.0
            

        du = np.linspace(z1u, self.length+z2u, numSegments)
        dd = np.linspace(z1d, self.length+z2d, numSegments)        
        au = self.width(du)
        ad = self.width(dd)
        if mode == 'x':
            wu = au[0]*sigma
            wd = ad[0]*sigma            
        elif mode == 'y':
            wu = au[1]*sigma
            wd = ad[1]*sigma            
        else:
            wu = sigma*(au[0]+au[1])/2
            wd = sigma*(ad[0]+ad[1])/2            

        v = np.vstack((du,wu))
        v = optics.geometric.vector_rotation_2D(v, self.dirAngle)
        v = v + np.array([self.pos]).T
        cv.add_shape(draw.PolyLine(x=v[0,:], y=v[1,:]), layername=self.layer+"_width")

        v = np.vstack((dd,-wd))
        v = optics.geometric.vector_rotation_2D(v, self.dirAngle)
        v = v + np.array([self.pos]).T
        cv.add_shape(draw.PolyLine(x=v[0,:], y=v[1,:]), layername=self.layer+"_width")        


#}}}        

#{{{ drawWidthOld

    def drawWidthOld(self, dxf, sigma, mode):
        #Draw the width

        #Determine the number of line segments to draw
        zr = q2zr(self.qx)
        resolution = zr/10.0
        if resolution > self.length/10.0:
            resolution = self.length/10.0

        numSegments = int(self.length/resolution)

        if numSegments > 100:
            numSegments = 100

        d = np.linspace(0,self.length, numSegments)
        a = self.width(d)
        if mode == 'x':
            w = a[0]*sigma
        elif mode == 'y':
            w = a[1]*sigma
        else:
            w = sigma*(a[0]+a[1])/2

        v = np.vstack((d,w))
        v = optics.geometric.vector_rotation_2D(v, self.dirAngle)
        v = v + np.array([self.pos]).T
        dxf.append(sdxf.LwPolyLine(points=list(v.T), layer=self.layer+"_width"))

        v = np.vstack((d,-w))
        v = optics.geometric.vector_rotation_2D(v, self.dirAngle)
        v = v + np.array([self.pos]).T
        dxf.append(sdxf.LwPolyLine(points=list(v.T), layer=self.layer+"_width"))


#}}}

#}}}

#{{{ Notification Handlers

    def _dirAngle_changed(self, old, new):
        self.set(trait_change_notify=False,
                 dirVect=array([np.cos(self.dirAngle), np.sin(self.dirAngle)]))
        self.set(trait_change_notify=False,
                 dirAngle = np.mod(self.dirAngle, 2*pi))
#        self.dirVect = array([np.cos(self.dirAngle), np.sin(self.dirAngle)])
#        self.dirAngle = np.mod(self.dirAngle, 2*pi)
            
    def _dirVect_changed(self, old, new):
        #Normalize
        self.set(trait_change_notify=False,
                 dirVect = self.dirVect/np.linalg.norm(array(self.dirVect)))
        #Update dirAngle accordingly
        self.set(trait_change_notify=False,
                 dirAngle = np.mod(np.arctan2(self.dirVect[1],
                                              self.dirVect[0]), 2*pi))

    def _qx_changed(self, old, new):
        self.wx = q2w(self.qx, wl=self.wl/self.n)
        self.set(trait_change_notify=False,
                 qrx = self.qx/self.n)
        self.q = optimalMatching(self.qx, self.qy)[0]

    def _qy_changed(self, old, new):
        self.wy = q2w(self.qy, wl=self.wl/self.n)
        self.set(trait_change_notify=False,
                 qry = self.qy/self.n)
        self.q = optimalMatching(self.qx, self.qy)[0]
        
    def _qrx_changed(self, old, new):
        self.qx = self.qrx*self.n
        
    def _qry_changed(self, old, new):
        self.qy = self.qry*self.n

    def _n_changed(self, old, new):
        self.set(trait_change_notify=False,
                 qx = self.qrx*self.n)
        self.set(trait_change_notify=False,
                 qy = self.qry*self.n)

#}}}        

#}}}

#{{{ Functions to determine the cross points

def optFunForStartPointR(phi, Mrot, R, q0, k, sigma, side):
    '''
    A function to return the distance between the point on
    the spherical surface at an angle phi and the beam width
    at the same z.
    '''
    cp = np.cos(phi)
    sp = np.sin(phi)
    v = R*np.array([[1-cp], [-sp]])

    a = np.dot(Mrot, v)
    z = a[0]
    w1 = a[1]*side
    w2 = sigma*np.sqrt(-2.0/(k*np.imag(1/(q0+z))))
    return (w1-w2)[0]

def optimStartPointR(theta, R, q0, k, sigma):
    '''
    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    Mrot = np.array([[ct, -st], [st, ct]])
    
    phi1 = sopt.newton(optFunForStartPointR, 0, args=(Mrot, R, q0, k, sigma, 1))
    cp = np.cos(phi1)
    sp = np.sin(phi1)
    v = R*np.array([[1-cp], [-sp]])
    z1 = np.dot(Mrot, v)[0][0]

    phi2 = sopt.newton(optFunForStartPointR, 0, args=(Mrot, R, q0, k, sigma, -1))
    cp = np.cos(phi2)
    sp = np.sin(phi2)
    v = R*np.array([[1-cp], [-sp]])
    z2 = np.dot(Mrot, v)[0][0]
    
    return (z1,z2) 

def optFunForEndPointR(phi, Mrot, R, q0, k, sigma, side):
    '''
    A function to return the distance between the point on
    the spherical surface at an angle phi and the beam width
    at the same z.
    '''
    cp = np.cos(phi)
    sp = np.sin(phi)
    v = -R*np.array([[1 - cp], [-sp]])

    a = np.dot(Mrot, v)
    z = a[0]
    w1 = a[1]*side
    w2 = sigma*np.sqrt(-2.0/(k*np.imag(1/(q0+z))))
    return (w1-w2)[0]

def optimEndPointR(theta, R, q0, k, sigma):
    '''
    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    Mrot = np.array([[ct, -st], [st, ct]])
    
    phi1 = sopt.newton(optFunForEndPointR, 0, args=(Mrot, R, q0, k, sigma, 1))
    cp = np.cos(phi1)
    sp = np.sin(phi1)
    v = R*np.array([[cp - 1], [sp]])
    z1 = np.dot(Mrot, v)[0][0]

    phi2 = sopt.newton(optFunForEndPointR, 0, args=(Mrot, R, q0, k, sigma, -1))
    cp = np.cos(phi2)
    sp = np.sin(phi2)
    v = R*np.array([[cp - 1], [sp]])
    z2 = np.dot(Mrot, v)[0][0]
    
    return (z1,z2) 

def optFunForFlat(a, Mrot, q0, k, sigma, side):
    '''
    A function to return the distance between the point on
    the spherical surface at an angle phi and the beam width
    at the same z.
    '''
    v = np.array([[0],[a]])

    b = np.dot(Mrot, v)
    z = b[0]
    w1 = b[1]*side
    w2 = sigma*np.sqrt(-2.0/(k*np.imag(1/(q0+z))))
    return (w1-w2)[0]

def optimCrossPointFlat(theta, q0, k, sigma):
    '''
    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    Mrot = np.array([[ct, -st], [st, ct]])
    
    a1 = sopt.newton(optFunForFlat, 0, args=(Mrot, q0, k, sigma, 1))
    v = np.array([[0],[a1]])
    z1 = np.dot(Mrot, v)[0][0]

    a2 = sopt.newton(optFunForFlat, 0, args=(Mrot, q0, k, sigma, -1))
    v = np.array([[0],[a2]])
    z2 = np.dot(Mrot, v)[0][0]
    
    return (z1,z2) 

#}}}
