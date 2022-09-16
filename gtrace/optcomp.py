'''
Define optical components for gtrace.

'''
#{{{ Import modules
import numpy as np
pi = np.pi
array = np.array
sqrt = np.lib.scimath.sqrt
from numpy.linalg import norm

from traits.api import HasTraits, Int, Float, CFloat, CArray, List, Str

import gtrace.optics as optics
import gtrace.optics.geometric
from .unit import *
import copy
import gtrace.draw as draw

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
__version__ = "0.2.1"
__maintainer__ = "Yoichi Aso"
__email__ = "yoichi.aso@nao.ac.jp"
__status__ = "Beta"

#}}}

#{{{ Generic Optics Class

class Optics(HasTraits):
    '''
    A general optics class from which other specific
    optics classes are derived.

    Attributes
    ----------
    name : str
        Name of the optics.
    center : array
        Center position of the optics. array of shape(2,).
    rotationAngle : float
        This angle defines the orientation of the optics.
    '''
    name = Str()
    center = CArray(dtype=np.float64, shape=(2,))
    rotationAngle = CFloat(0.0) #in rad

#{{{ isHit(beam)

    def isHit(beam):
        '''
        A function to see if a beam hits this optics or not.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.

        Returns
        -------
        Dict
            The return value is a dictionary with the following keys:
            ``isHit, position, distance, face``

            ``isHit``:
            This is a boolean to answer whether the beam hit the optics
            or not.

            ``position``:
            A numpy array containing the coordinate values of the intersection
            point between the beam and the optics. If isHit is False, this parameter
            does not mean anything.

            ``distance``
            The distance between the beam origin and the intersection point.

            ``face``:
            An optional string identifying which face of the optics was hit.
            For example, ``face`` can be either "HR" or "AR" for a mirror.
            ``face`` can also be "side", meaning that the beam hits a side
            of the optics, which is not meant to be used, e.g. the side of a mirror.
            In this case, the beam have reached a dead end.
        '''
        #This is an abstract function
        return {'isHit': False, 'position': np.array((0,0)),
                'distance': 0.0, 'face':''}

#}}}

#{{{ hit(beam, order=0, threshold=0.0):

    def hit(beam, order=0, threshold=0.0):
        '''
        A function to hit the optics with a beam.

        This function attempts to hit the optics with the source beam, ``beam``.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.

        Returns
        -------
        {boolean, dict, str}
            ``(isHit, beamDict, face)``

            ``isHit``
            This is a boolean to answer whether the beam hit the optics
            or not.

            ``beamDict``
            A dictionary containing resultant beams.

            ``face``:
              An optional string identifying which face of the optics was hit.
              For a mirror, ``face`` is any of "HR", "AR" or "side".
        '''
        #This is an abstract function
        return {False, {}, "side"}  # Is this a bug? Shouldn't it be a tuple?

#}}}

#{{{  _isHitSurface_()

    def _isHitSurface_(self, beam, surface_center, normal_vector,
                       surface_size=1.0, inv_ROC=0.0):
        '''
        Determine if a beam hit a surface

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.

        Returns
        -------
        ans : dict
            A dictionary with the following keys:
            "isHit": A boolean value whether the beam hit the surface or not.
            "Intersection Point": numpy array of the coordinates of the intersection point.
            "distance": Distance between the origin of the beam and the intersection point.
            "localNormVect": A numpy array representing the normal vector of the surface
                                        at the intersection point.
            "localNormAngle": The angle of the localNormVect.
        '''
        if np.abs(inv_ROC) < 1e-5:

            ans = optics.geometric.line_plane_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                          plane_center=surface_center, normalVector=normal_vector,
                                          diameter=surface_size)
            localNormVect = normal_vector
            localNormAngle = np.mod(np.arctan2(localNormVect[1],
                                               localNormVect[0]), 2*pi)
            ans['localNormVect'] = localNormVect
            ans['localNormAngle'] = localNormAngle
            return ans
        else:
            ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                         chord_center=surface_center,
                                                         chordNormVect=normal_vector,
                                                         invROC=inv_ROC,
                                                         diameter=surface_size)

            return ans

#}}}

#}}}

#{{{ Mirror Class

class Mirror(Optics):
    '''
    Representing a partial reflective mirror.

    Attributes
    ----------
    curve_direction : str
        Either 'h' or 'v'. If it is 'h' the mirror is curved in horizontal plane. If 'v', it is vertical.
    HRcenter : array
        The position of the center of the arc of the HR surface. shape(2,).
    HRcenterC : array
        The position of the center of the chord of the HR surface. shape(2,).
    normVectHR : array
        Normal vector of the HR surface. shape(2,)
    normAngleHR : float
        Angle of the HR normal vector. In radians.
    ARcenter : array
        The position of the center of the AR surface. shape(2,)
    normVectAR : array
        Normal vector of the HR surface. shape(2,)
    normAngleAR : float
        Angle of the HR normal vector. In radians.
    HRtransmissive : boolean
        A boolean value defaults to False. If True, this mirror
        is supposed to transmit beams on the HR surface. Therefore,
        for the first encounter of a beam on the HR surface of this mirror
        will not increase the stray_order. This flag should be set to True for
        beam splitters and input test masses.
    term_on_HR : boolean
        If this is True, a beam with stray_order <= self.term_on_HR_order will be terminated when
        it hits on HR. This is to avoid the inifinite loop of non-sequencial
        trace by forming a cavity.
    term_on_HR_order : int
        Integer to specify the upper limit of the stray order used to judge
        whether to terminate the non sequential trace or not on HR reflection.
    '''

#{{{ Traits definitions

    HRcenter = CArray(dtype=np.float64, shape=(2,))
    HRcenterC = CArray(dtype=np.float64, shape=(2,))
    sagHR = CFloat()
    normVectHR = CArray(dtype=np.float64, shape=(2,))
    normAngleHR = CFloat()

    ARcenter = CArray(dtype=np.float64, shape=(2,))
    ARcenterC = CArray(dtype=np.float64, shape=(2,))
    sagAR = CFloat()
    normVectAR = CArray(dtype=np.float64, shape=(2,))
    normAngleAR = CFloat()

    diameter = CFloat(25.0*cm) #
    ARdiameter = CFloat()
    thickness = CFloat(15.0*cm) #
    wedgeAngle = CFloat(0.25*pi/180) # in rad
    n = CFloat(1.45) #Index of refraction

    inv_ROC_HR = CFloat(1.0/7000.0) #Inverse of the ROC of the HR surface.
    inv_ROC_AR = CFloat(0.0) #Inverse of the ROC of the AR surface.

    Refl_HR = CFloat(99.0) #Power reflectivity of the HR side.
    Trans_HR = CFloat(1.0) #Power transmittance of the HR side.

    Refl_AR = CFloat(0.01) #Power reflectivity of the AR side.
    Trans_AR = CFloat(99.99) #Power transmittance of the HR side.


#}}}

#{{{ __init__

    def __init__(self, HRcenter=[0.0,0.0], normAngleHR=0.0,
                 normVectHR=None, diameter=25.0*cm, thickness=15.0*cm,
                 wedgeAngle=0.25*pi/180., inv_ROC_HR=1.0/7000.0, inv_ROC_AR=0.0,
                 Refl_HR=0.99, Trans_HR=0.01, Refl_AR=0.01, Trans_AR=0.99, n=1.45,
                 name="Mirror", HRtransmissive=False, term_on_HR=False):
        '''
        Create a mirror object.

        Parameters
        ----------
        HRcenter : array, optional
            Position of the center of the HR surface.
            Defaults [0.0, 0.0].
        normAngleHR : float, optional
            Direction angle of the normal vector of the HR surface. In radians.
            Defaults 0.0.
        normVectHR : arrary or None, optional
            Normal vector of the HR surface. Should be an array of shape(2,).
            Defaults None.
        diameter : float, optional
            Diameter of the mirror.
            Defaults 25.0*cm.
        thickness : float, optional
            Thickness of the mirror.
            Defaults 15.0*cm.
        wedgeAngle : float, optional
            Wedge angle between the HR and AR surfaces. In radians.
            Defaults 0.25*pi/180.
        inv_ROC_HR : float, optional
            1/ROC of the HR surface.
            Defaults 1.0/7000.0.
        inv_ROC_AR : float, optional
            1/ROC of the AR surface.
            Defaults 0.0.
        Refl_HR : float, optional
            Power reflectivity of the HR surface.
            Defaults 0.99.
        Trans_HR : float, optional
            Power transmissivity of the HR surface.
            Defaults 0.01.
        Refl_AR : float, optional
            Power reflectivity of the AR surface.
            Defaults 0.01.
        Trans_AR : float, optional
            Power transmissivity of the AR surface.
            Defaults 0.99.
        n : float, optional
            Index of refraction.
            Defaults 1.45.
        name : str, optional
            Name of the mirror.
            Defaults "Mirror".
        HRtransmissive : boolean, optional
            If True, this mirror
            is supposed to transmit beams on the HR surface. Therefore,
            for the first encounter of a beam on the HR surface of this mirror
            will not increase the stray_order. This flag should be set to True for
            beam splitters and input test masses.
            Defaults False
        term_on_HR : boolean, optional
            If this is True, a beam with stray_order <= self.term_on_HR_order
            will be terminated when
            it hits on HR. This is to avoid the inifinite loop of
            non-sequencial
            trace by forming a cavity.
            Defaults False.
        '''
        self.diameter = diameter

        #Compute the sag.
        #Sag is positive for convex mirror.
        if np.abs(inv_ROC_HR) > 1./(10*km):
            R = 1./inv_ROC_HR
            r = self.diameter/2
            self.sagHR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagHR = 0.0;

        #Convert rotationAngle to normVectHR or vice versa.
        if normVectHR is not None:
            self.normVectHR = normVectHR
        else:
            self.normAngleHR = normAngleHR

        self.HRcenter = HRcenter
        self._HRcenter_changed(0,0)

        self.thickness = thickness
        self.wedgeAngle = wedgeAngle
        self.ARdiameter = self.diameter/np.cos(self.wedgeAngle)
        self.inv_ROC_HR = inv_ROC_HR
        self.inv_ROC_AR = inv_ROC_AR
        self.Refl_HR = Refl_HR
        self.Trans_HR = Trans_HR
        self.Refl_AR = Refl_AR
        self.Trans_AR = Trans_AR
        self.n = n
        self._normAngleHR_changed(0,0)
        self.name = name
        self.HRtransmissive = HRtransmissive
        self.term_on_HR = term_on_HR
        self.term_on_HR_order = 0

#}}}

#{{{ copy

    def copy(self):
        return Mirror(HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                      diameter=self.diameter, thickness=self.thickness,
                      wedgeAngle=self.wedgeAngle, inv_ROC_HR=self.inv_ROC_HR,
                      inv_ROC_AR=self.inv_ROC_AR, Refl_HR=self.Refl_HR,
                      Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR, Trans_AR=self.Trans_AR,
                      n=self.n, name=self.name, HRtransmissive=self.HRtransmissive,
                      term_on_HR=self.term_on_HR)

#}}}

#{{{ get_side_info

    def get_side_info(self):
        '''
        Return information on the sides of the mirror.
        Returned value is a list of two tuples like [(center1, normVect1, length1), (center2, normVect2, length2)]
        Each tuple corresponds to a side. center1 is the coordinates of the center of the side line. normVect1 is the normal vector of the side line. length1 is the length of the side line.

        Returns
        -------
        [(float, float, float), (float, float, float)]
        '''

        r = self.diameter/2

        v1h = np.array([self.thickness/2, r])
        v1a = np.array([-self.thickness/2 - r*np.tan(self.wedgeAngle), r])
        v1h = optics.geometric.vector_rotation_2D(v1h, self.normAngleHR) + self.center
        v1a = optics.geometric.vector_rotation_2D(v1a, self.normAngleHR) + self.center

        center1 = (v1h + v1a)/2
        vn1 = optics.geometric.vector_rotation_2D(v1h - v1a, pi/2)
        normVect1 = vn1/np.linalg.norm(vn1)
        length1 = np.linalg.norm(v1h - v1a)

        v2h = np.array([self.thickness/2, -r])
        v2a = np.array([-self.thickness/2 + r*np.tan(self.wedgeAngle), -r])
        v2h = optics.geometric.vector_rotation_2D(v2h, self.normAngleHR) + self.center
        v2a = optics.geometric.vector_rotation_2D(v2a, self.normAngleHR) + self.center

        center2 = (v2h + v2a)/2
        vn2 = optics.geometric.vector_rotation_2D(v2h - v2a, -pi/2)
        normVect2 = vn2/np.linalg.norm(vn2)
        length2 = np.linalg.norm(v2h - v2a)

        return [(center1, normVect1, length1), (center2, normVect2, length2)]



#}}}

#{{{ rotate

    def rotate(self, angle, center=False):
        '''
        Rotate the mirror. If center is not specified, the center of rotation is
        HRcenter. If center is given (as a vector), the center of rotation is
        center. center is a position vector in the global coordinates.

        Parameters
        ----------
        angle : float
            Angle of rotation.
        center: array or boolean, optional
            Center of rotation, or False.
        '''
        if center is not False:
            center = np.array(center)
            pointer = self.HRcenter - center
            pointer = optics.geometric.vector_rotation_2D(pointer, angle)
            self.HRcenter = center + pointer

        self.normAngleHR = self.normAngleHR + angle
#}}}

#{{{ Translate

    def translate(self, trVect):
        trVect = np.array(trVect)
        self.center = self.center + trVect

#}}}

#{{{ Draw

    def draw(self, cv, drawName=False):
        '''
        Draw itself
        '''

        plVect = optics.geometric.vector_rotation_2D(self.normVectHR, pi/2)
        p1 = self.HRcenterC + plVect * self.diameter/2
        p2 = p1 - plVect * self.diameter
        p3 = p2 - self.normVectHR * (self.thickness - np.tan(self.wedgeAngle)*self.diameter/2)
        p4 = p1 - self.normVectHR * (self.thickness + np.tan(self.wedgeAngle)*self.diameter/2)

        cv.add_shape(draw.Line(p2,p3), layername="Mirrors")
        cv.add_shape(draw.Line(p4,p1), layername="Mirrors")

        d = self.thickness/10
        l1 = p1 - self.normVectHR * d
        l2 = p2 - self.normVectHR * d
        cv.add_shape(draw.Line(l1,l2), layername="Mirrors")

        #Draw Curved surface

        #HR

        if np.abs(self.inv_ROC_HR) > 1.0/1e5:
            R = 1/self.inv_ROC_HR
            theta = np.arcsin(self.diameter/2/R)
            sag = R*(1-np.cos(theta))
            x = np.linspace(0, self.diameter/2, 30)
            y = R*(1.0 - np.sqrt(1.0 - x**2/(R**2))) -sag
            x2 = -np.flipud(x)
            y2 = np.flipud(y)
            x = np.hstack((x2,x))
            y = np.hstack((y2,y))
            v = np.vstack((x,y))
            v = optics.geometric.vector_rotation_2D(v, self.normAngleHR - pi/2)
            v = v.T + self.HRcenterC
            cv.add_shape(draw.PolyLine(x=v[:,0], y=v[:,1]), layername="Mirrors")
            #dxf.append(sdxf.LwPolyLine(points=list(v), layer="Mirrors"))
        else:
            cv.add_shape(draw.Line(p1,p2), layername="Mirrors")
            #dxf.append(sdxf.Line(points=[p1,p2], layer="Mirrors"))

        #AR
        if np.abs(self.inv_ROC_AR) > 1.0/1e5:
            diameter = self.diameter/np.cos(self.wedgeAngle)

            R = 1/self.inv_ROC_AR
            theta = np.arcsin(diameter/2/R)
            sag = R*(1-np.cos(theta))
            x = np.linspace(0, diameter/2, 30)
            y = R*(1.0 - np.sqrt(1.0 - x**2/(R**2))) -sag
            x2 = -np.flipud(x)
            y2 = np.flipud(y)
            x = np.hstack((x2,x))
            y = np.hstack((y2,y))
            v = np.vstack((x,y))
            v = optics.geometric.vector_rotation_2D(v, self.normAngleAR - pi/2)
            v = v.T + self.ARcenter
            cv.add_shape(draw.PolyLine(x=v[:,0], y=v[:,1]), layername="Mirrors")
            #dxf.append(sdxf.LwPolyLine(points=list(v), layer="Mirrors"))
        else:
            cv.add_shape(draw.Line(p3,p4), layername="Mirrors")
            #dxf.append(sdxf.Line(points=[p3,p4], layer="Mirrors"))


        if drawName:
            center = (p1+p2+p3+p4)/4.
            height = self.thickness/4.
            width = height*len(self.name)
            center = center - np.array([width/2, height/2])
            cv.add_shape(draw.Text(text=self.name, point=center,height=height),
                         layername="text")
            # dxf.append(sdxf.Text(text=self.name, point=center, #
            #                      height=height, layer='text'))


#}}}

#{{{ isHit

    def isHit(self, beam):
        '''
        A function to see if a beam hits this optics or not.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.

        Returns
        -------
        Dict
            The return value is a dictionary with the following keys:
            ``isHit, position, distance, face``

            ``isHit``:
            This is a boolean to answer whether the beam hit the optics
            or not.

            ``position``:
            A numpy array containing the coordinate values of the intersection
            point between the beam and the optics. If isHit is False, this parameter
            does not mean anything.

            ``distance``
            The distance between the beam origin and the intersection point.

            ``face``:
            An optional string identifying which face of the optics was hit.
            For example, ``face`` can be either "HR" or "AR" for a mirror.
            ``face`` can also be "side", meaning that the beam hits a side
            of the optics, which is not meant to be used, e.g. the side of a mirror.
            In this case, the beam have reached a dead end.
        '''

        HRsurface = {'center': self.HRcenterC, 'normal_vector': self.normVectHR,
                     'size': self.diameter, 'inv_ROC': self.inv_ROC_HR, 'name': 'HR'}
        ARsurface = {'center': self.ARcenter, 'normal_vector': self.normVectAR,
                     'size': self.diameter, 'inv_ROC': 0.0, 'name': 'AR'}

        # #The vector parallel to the HR surface, pointing left.
        # v1 = np.array((-self.normVectHR[1], self.normVectHR[0]))
        # #Left corner of the HR surface
        # c1 = self.HRcenterC + self.diameter/2 * v1
        # #Right corner of the HR surface
        # c2 = self.HRcenterC - self.diameter/2 * v1
        # #Center of the side 1
        # side_center_1 = c1 + self.thickness/2 * (- self.normVectHR)
        # #Center of the side 2
        # side_center_2 = c2 + self.thickness/2 * (- self.normVectHR)

        # Side2 = {'center': side_center_2, 'normal_vector': -v1,
        #              'size': self.thickness, 'inv_ROC': 0.0, 'name': 'side'}

        sides = self.get_side_info()

        Side1 = {'center': sides[0][0], 'normal_vector': sides[0][1],
                     'size': sides[0][2], 'inv_ROC': 0.0, 'name': 'side'}
        Side2 = {'center': sides[1][0], 'normal_vector': sides[1][1],
                     'size': sides[1][2], 'inv_ROC': 0.0, 'name': 'side'}

        faceList = [HRsurface, ARsurface, Side1, Side2]

        min_dist = 1e16
        final_answer = None
        for face in faceList:
            ans = self._isHitSurface_(beam, surface_center=face['center'],
                                normal_vector=face['normal_vector'],
                                surface_size=face['size'], inv_ROC=face['inv_ROC'])
            if ans['isHit']:
                if min_dist > ans['distance']:
                    min_dist = ans['distance']
                    final_answer = ans
                    face_name = face['name']

        if final_answer is None:
            return {'isHit': False, 'position': np.array((0,0)),
                'distance': 0.0, 'face':''}
        else:
            return {'isHit': True, 'position': final_answer['Intersection Point'],
                'distance': min_dist, 'face': face_name}



#}}}

#{{{ hit()
    def hit(self, beam, order=0, threshold=0.0, face=False):
        '''
        A function to hit the optics with a beam.

        This function attempts to hit the optics with the source beam, ``beam``.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.

        Returns
        -------
        {boolean, dict, str}
            ``(isHit, beamDict, face)``

            ``isHit``
            This is a boolean to answer whether the beam hit the optics
            or not.

            ``beamDict``
            A dictionary containing resultant beams.

            ``face``:
              An optional string identifying which face of the optics was hit.
              For a mirror, ``face`` is any of "HR", "AR" or "side".
        '''

        #If an optional argument ``face`` is specified
        if face:
            if face == 'HR':
                beams = self.hitFromHR(beam, order=order, threshold=threshold)
            elif face == 'AR':
                beams = self.hitFromAR(beam, order=order, threshold=threshold)
            else:
                print(('Wrong face %s is specified'%face))
                return (False, {}, "")
        #If face is not specified
        else:
            #Check if the beam hit the mirror
            ans = self.isHit(beam)
            face = ans['face']
            if ans['isHit']:
                if face == 'HR':
                    beams = self.hitFromHR(beam, order=order, threshold=threshold)
                elif face == 'AR':
                    beams = self.hitFromAR(beam, order=order, threshold=threshold)
                else:
                    #The beam hit a side of the mirror
                    inputBeam = beam.copy()
                    inputBeam.length=ans['distance']
                    return (True, {inputBeam}, "side")
            else:
                #The beam did not hit the mirror
                return (False, {}, "")

        return (True, beams, face)


#}}}

#{{{ hitFromHR

    def hitFromHR(self, beam, order=0, threshold=0.0, verbose=False):
        '''
        Compute the reflected and deflected beams when
        an input beam hit the HR surface.

        The internal reflections are computed as long as the number
        of internal reflections are below the ``order`` and the power
        of the reflected beams is over the threshold.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.
        verbose : boolean, optional
            Print useful information.

        Returns
        -------
        beams : dict
            Dictionary of reflected and deflected beams.
        '''

        #A dictionary to hold beams
        beams={}

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=self.HRcenterC,
                                                     chordNormVect=self.normVectHR,
                                                     invROC=self.inv_ROC_HR,
                                                     diameter=self.diameter)
        if not ans['isHit']:
            #The input beam does not hit the mirror.
            if verbose:
                print((self.name + ': The beam does not hit the mirror'))
            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_in = beam.copy() #Make a copy
        beam_in.length = ans['distance']
        beam_in.incSurfAngle = localNormAngle
        beam_in.incSurfInvROC = self.inv_ROC_HR
        beams['input']= beam_in


        #Propagate the input beam to the intersection point
        beam_on_HR = beam_in.copy()
        beam_on_HR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                               localNormAngle,
                                               1.0, self.n, invROC=self.inv_ROC_HR)
        #Reflected beam
        beam_r1 = beam_on_HR.copy()
        beam_r1.P = beam_r1.P * self.Refl_HR
        if beam_r1.P > threshold:
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = self.inv_ROC_HR
            beam_r1.incSurfAngle = None
            beam_r1.incSurfInvROC = None
            beam_r1.name = self.name+':r1'
            beams['r1'] = beam_r1

        #Transmitted beam
        beam_s1 = beam_on_HR.copy()
        beam_s1.P = beam_s1.P * self.Trans_HR
        if not self.HRtransmissive:
            beam_s1.stray_order = beam_s1.stray_order+1
        if beam_s1.P < threshold or beam_s1.stray_order > order:
            return beams
        beam_s1.dirAngle = deflAngle
        beam_s1.n = self.n
        beam_s1.ABCDTrans(Mtx, Mty)
        beam_s1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
        beam_s1.departSurfInvROC = -self.inv_ROC_HR

        #Hit AR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=self.ARcenter,
                                                     chordNormVect=-self.normVectAR,
                                                     invROC=-self.inv_ROC_AR,
                                                     diameter=self.ARdiameter)

        if not ans['isHit']:
            #The beam does not hit the AR surface. It must hit either of the sides.

            #Get side information
            sides = self.get_side_info()

            #Loop for sides
            for side in sides:
                ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                if ans['isHit']:
                    localNormAngle = ans['localNormAngle']
                    beam_s1.length = ans['distance']
                    beam_s1.layer = 'aux_beam'
                    beam_s1.incSurfAngle = localNormAngle
                    beam_s1.incSurfInvROC = 0.0
                    beam_s1.name = self.name+':s1'
                    beams['s1']= beam_s1
                    return beams

            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_s1.length = ans['distance']
        beam_s1.incSurfAngle = localNormAngle
        beam_s1.incSurfInvROC = -self.inv_ROC_AR
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the AR surface
        beam_on_AR = beam_s1.copy()
        beam_on_AR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                               localNormAngle,
                                               self.n, 1.0, invROC=-self.inv_ROC_AR)

        #Transmitted beam
        beam_t1 = beam_on_AR.copy()
        beam_t1.P = beam_on_AR.P * self.Trans_AR
        if beam_t1.P > threshold:
            beam_t1.dirAngle = deflAngle
            beam_t1.n = 1.0
            beam_t1.ABCDTrans(Mtx, Mty)
            beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
            beam_t1.departSurfInvROC = self.inv_ROC_AR
            beam_t1.incSurfAngle = None
            beam_t1.incSurfInvROC = None
            beam_t1.name = self.name+':t%d'%(1)
            beams['t1'] = beam_t1


        #Reflected beam
        beam_sr = beam_on_AR.copy()
        beam_sr.P = beam_sr.P * self.Refl_AR
        beam_sr.stray_order = beam_sr.stray_order+1
        if beam_sr.P < threshold or beam_sr.stray_order > order:
            return beams
        beam_sr.dirAngle = reflAngle
        beam_sr.ABCDTrans(Mrx, Mry)
        beam_sr.departSurfAngle = localNormAngle
        beam_sr.departSurfInvROC = -self.inv_ROC_AR


        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit the HR from the back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=self.HRcenterC,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-self.inv_ROC_HR,
                                                         diameter=self.diameter)


            if not ans['isHit']:
                #The beam does not hit the HR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_sr.length = ans['distance']
                        beam_sr.layer = 'aux_beam'
                        beam_sr.incSurfAngle = localNormAngle
                        beam_sr.incSurfInvROC = 0.0
                        beam_sr.name = self.name+':s%d'%(2*ii)
                        beams['s'+str(2*ii)]= beam_sr
                        break

                break

            #Local normal angle
            localNormAngle = ans['localNormAngle']

            beam_sr.length = ans['distance']
            beam_sr.layer = 'aux_beam'
            beam_sr.incSurfAngle = localNormAngle
            beam_sr.incSurfInvROC = -self.inv_ROC_HR
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr

            #Propagate the input beam to the intersection point
            beam_on_HR = beam_sr.copy()
            beam_on_HR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-self.inv_ROC_HR)

            #Transmitted through HR
            beam_r1 = beam_on_HR.copy()
            beam_r1.P = beam_r1.P * self.Trans_HR
            beam_r1.stray_order = beam_r1.stray_order+1
            if beam_r1.P > threshold and beam_r1.stray_order <= order:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = self.inv_ROC_HR
                beam_r1.incSurfAngle = None
                beam_r1.incSurfInvROC = None
                beam_r1.name = self.name+':r%d'%(ii+1)
                beams['r'+str(ii+1)] = beam_r1

            #Reflected by HR
            beam_s1 = beam_on_HR.copy()
            beam_s1.P = beam_s1.P * self.Refl_HR
            if beam_s1.P < threshold:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.P = beam_s1.P * self.Refl_HR
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -self.inv_ROC_HR

            #Hit AR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=self.ARcenter,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-self.inv_ROC_AR,
                                                         diameter=self.ARdiameter)

            if not ans['isHit']:
                #The beam does not hit the AR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_s1.length = ans['distance']
                        beam_s1.layer = 'aux_beam'
                        beam_s1.incSurfAngle = localNormAngle
                        beam_s1.incSurfInvROC = 0.0
                        beam_s1.name = self.name+':s%d'%(2*ii+1)
                        beams['s'+str(2*ii+1)]= beam_s1
                        break

                break


            #Local normal angle
            localNormAngle = ans['localNormAngle']

            beam_s1.incSurfAngle = localNormAngle
            beam_s1.incSurfInvROC = -self.inv_ROC_AR
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1

            #Propagate the beam to the AR surface
            beam_on_AR = beam_s1.copy()
            beam_on_AR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-self.inv_ROC_AR)
            #Transmitted beam
            beam_t1 = beam_on_AR.copy()
            beam_t1.P = beam_on_AR.P * self.Trans_AR
            if beam_t1.P > threshold:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = self.inv_ROC_AR
                beam_t1.incSurfAngle = None
                beam_t1.incSurfInvROC = None
                beam_t1.name = self.name+':t%d'%(ii+1)
                beams['t'+str(ii+1)] = beam_t1

            #Reflected beam
            beam_sr = beam_on_AR.copy()
            beam_sr.P = beam_sr.P * self.Refl_AR
            beam_sr.stray_order = beam_sr.stray_order+1
            if beam_sr.P < threshold or beam_sr.stray_order > order:
                break
            beam_sr.dirAngle = reflAngle
            beam_sr.ABCDTrans(Mrx, Mry)
            beam_sr.departSurfAngle = localNormAngle
            beam_sr.departSurfInvROC = -self.inv_ROC_AR

            ii=ii+1

        return beams
#}}}

#{{{ hitFromAR

    def hitFromAR(self, beam, order=0, threshold=0.0, verbose=False):
        '''
        Compute the reflected and deflected beams when
        an input beam hit the AR surface.

        The internal reflections are computed as long as the number
        of internal reflections are below the ``order`` and the power
        of the reflected beams is over the threshold.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.
        verbose : boolean, optional
            Print useful information.

        Returns
        -------
        beams : dict
            Dictionary of reflected and deflected beams.
        '''

        #A dictionary to hold beams
        beams={}

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=self.ARcenter,
                                                     chordNormVect=self.normVectAR,
                                                     invROC=self.inv_ROC_AR,
                                                     diameter=self.ARdiameter)

        if not ans['isHit']:
            #The input beam does not hit the mirror.
            if verbose:
                print((self.name + ': The beam does not hit the mirror'))
            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_in = beam.copy() #Make a copy
        beam_in.incSurfAngle = localNormAngle
        beam_in.incSurfInvROC = self.inv_ROC_AR
        beam_in.length = ans['distance']
        beams['input']= beam_in

        #Propagate the input beam to the intersection point
        beam_on_AR = beam_in.copy()
        beam_on_AR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                               localNormAngle,
                                               1.0, self.n, invROC=self.inv_ROC_AR)
        #Reflected beam
        beam_r1 = beam_on_AR.copy()
        beam_r1.P = beam_r1.P * self.Refl_AR
        beam_r1.stray_order = beam_r1.stray_order+1
        if beam_r1.P > threshold and beam_r1.stray_order <= order:
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = self.inv_ROC_AR
            beam_r1.incSurfAngle = None
            beam_r1.incSurfInvROC = None
            beam_r1.name = self.name+':r1'
            beams['r1'] = beam_r1

        #Transmitted beam
        beam_s1 = beam_on_AR.copy()
        beam_s1.P = beam_s1.P * self.Trans_AR
        if beam_s1.P < threshold:
            return beams
        beam_s1.dirAngle = deflAngle
        beam_s1.n = self.n
        beam_s1.ABCDTrans(Mtx, Mty)
        beam_s1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
        beam_s1.departSurfInvROC = -self.inv_ROC_AR

        #Hit HR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=self.HRcenterC,
                                                     chordNormVect=-self.normVectHR,
                                                     invROC=-self.inv_ROC_HR,
                                                     diameter=self.diameter)

        if not ans['isHit']:
            #The beam does not hit the HR surface. It must hit either of the sides.

            #Get side information
            sides = self.get_side_info()

            #Loop for sides
            for side in sides:
                ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                if ans['isHit']:
                    localNormAngle = ans['localNormAngle']
                    beam_s1.length = ans['distance']
                    beam_s1.layer = 'aux_beam'
                    beam_s1.incSurfAngle = localNormAngle
                    beam_s1.incSurfInvROC = 0.0
                    beam_s1.name = self.name+':s1'
                    beams['s1']= beam_s1
                    return beams

            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']
        beam_s1.length = ans['distance']
        beam_s1.incSurfAngle = localNormAngle
        beam_s1.incSurfInvROC = -self.inv_ROC_HR
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the HR surface
        beam_on_HR = beam_s1.copy()
        beam_on_HR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                               localNormAngle,
                                               self.n, 1.0, invROC=-self.inv_ROC_HR)

        #Transmitted beam
        beam_t1 = beam_on_HR.copy()
        beam_t1.P = beam_on_HR.P * self.Trans_HR
        if not self.HRtransmissive:
            beam_t1.stray_order = beam_t1.stray_order+1
        if beam_t1.P > threshold and beam_t1.stray_order <= order:
            beam_t1.dirAngle = deflAngle
            beam_t1.n = 1.0
            beam_t1.ABCDTrans(Mtx, Mty)
            beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
            beam_t1.departSurfInvROC = self.inv_ROC_HR
            beam_t1.incSurfAngle = None
            beam_t1.incSurfInvROC = None
            beam_t1.name = self.name+':t1'
            beams['t1'] = beam_t1

        #Reflected beam
        beam_sr = beam_on_HR.copy()
        beam_sr.P = beam_sr.P * self.Refl_HR
        if beam_sr.P < threshold:
            return beams
        beam_sr.dirAngle = reflAngle
        beam_sr.ABCDTrans(Mrx, Mry)
        beam_sr.departSurfAngle = localNormAngle
        beam_sr.departSurfInvROC = -self.inv_ROC_HR

        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit AR from back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=self.ARcenter,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-self.inv_ROC_AR,
                                                         diameter=self.ARdiameter)

            if not ans['isHit']:
                #The beam does not hit the AR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_sr.length = ans['distance']
                        beam_sr.layer = 'aux_beam'
                        beam_sr.incSurfAngle = localNormAngle
                        beam_sr.incSurfInvROC = 0.0
                        beam_sr.name = self.name+':s%d'%(2*ii)
                        beams['s'+str(2*ii)]= beam_sr
                        break

                break

            #Local normal angle
            localNormAngle = ans['localNormAngle']
            beam_sr.length = ans['distance']
            beam_sr.layer = 'aux_beam'
            beam_sr.incSurfAngle = localNormAngle
            beam_sr.incSurfInvROC = -self.inv_ROC_AR
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr


            #Propagate the input beam to the intersection point
            beam_on_AR = beam_sr.copy()
            beam_on_AR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-self.inv_ROC_AR)

            #Transmitted through AR
            beam_r1 = beam_on_AR.copy()
            beam_r1.P = beam_r1.P * self.Trans_AR
            if beam_r1.P > threshold:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = self.inv_ROC_AR
                beam_r1.incSurfAngle = None
                beam_r1.incSurfInvROC = None
                beam_r1.name = self.name+':r%d'%(ii+1)
                beams['r'+str(ii+1)] = beam_r1

            #Reflected by AR
            beam_s1 = beam_on_AR.copy()
            beam_s1.P = beam_s1.P * self.Refl_AR
            beam_s1.stray_order = beam_s1.stray_order+1
            if beam_s1.P < threshold or beam_s1.stray_order > order:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -self.inv_ROC_AR

            #Hit HR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=self.HRcenterC,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-self.inv_ROC_HR,
                                                         diameter=self.diameter)

            if not ans['isHit']:
                #The beam does not hit the HR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_s1.length = ans['distance']
                        beam_s1.layer = 'aux_beam'
                        beam_s1.incSurfAngle = localNormAngle
                        beam_s1.incSurfInvROC = 0.0
                        beam_s1.name = self.name+':s%d'%(2*ii+1)
                        beams['s'+str(2*ii+1)]= beam_s1
                        break

                break

           #Local normal angle
            localNormAngle = ans['localNormAngle']
            beam_s1.incSurfAngle = localNormAngle
            beam_s1.incSurfInvROC = -self.inv_ROC_HR
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1


            #Propagate the beam to the HR surface
            beam_on_HR = beam_s1.copy()
            beam_on_HR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-self.inv_ROC_HR)

            #Transmitted beam
            beam_t1 = beam_on_HR.copy()
            beam_t1.P = beam_t1.P * self.Trans_HR
            beam_t1.stray_order = beam_t1.stray_order+1
            if beam_t1.P > threshold and beam_t1.stray_order <= order:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = self.inv_ROC_HR
                beam_t1.incSurfAngle = None
                beam_t1.incSurfInvROC = None
                beam_t1.name = self.name+':t%d'%(ii+1)
                beams['t'+str(ii+1)] = beam_t1

            #Reflected beam
            beam_sr = beam_on_HR.copy()
            beam_sr.P = beam_sr.P * self.Refl_HR
            if beam_sr.P < threshold:
                break
            beam_sr.dirAngle = reflAngle
            beam_sr.ABCDTrans(Mrx, Mry)
            beam_sr.departSurfAngle = localNormAngle
            beam_sr.departSurfInvROC = -self.inv_ROC_HR

            ii=ii+1

        return beams

#}}}

#{{{ Notification handlers

    def _normAngleHR_changed(self, old, new):
        self.set(trait_change_notify=False,
                 normVectHR = array([np.cos(self.normAngleHR), np.sin(self.normAngleHR)]))
        self.set(trait_change_notify=False,
                 normAngleHR = np.mod(self.normAngleHR, 2*pi))

        self.normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle)
        self.normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi)
        self.HRcenterC = self.HRcenter - self.normVectHR * self.sagHR

    def _normVectHR_changed(self, old, new):
        #Normalize
        self.set(trait_change_notify=False,
                 normVectHR = self.normVectHR/np.linalg.norm(array(self.normVectHR)))
        #Update dirAngle accordingly
        self.set(trait_change_notify=False,
                 normAngleHR = np.mod(np.arctan2(self.normVectHR[1],
                                                   self.normVectHR[0]), 2*pi))

        self.normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle)
        self.normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi)
        self.HRcenterC = self.HRcenter - self.normVectHR * self.sagHR

    def _HRcenterC_changed(self, old, new):
        self.set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)
        self.set(trait_change_notify=False,
                 center = (self.HRcenterC + self.ARcenter)/2.0)
        self.set(trait_change_notify=False,
                 HRcenter = self.HRcenterC + self.sagHR*self.normVectHR)

    def _HRcenter_changed(self, old, new):
        self.set(trait_change_notify=False,
                 HRcenterC = self.HRcenter - self.sagHR*self.normVectHR)
        self.set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)
        self.set(trait_change_notify=False,
                 center = (self.HRcenterC + self.ARcenterC)/2.0)

    def _center_changed(self, old, new):
        self.set(trait_change_notify=False,
                 HRcenterC = self.center + self.normVectHR * self.thickness/2.0)
        self.set(trait_change_notify=False,
                 HRcenter = self.HRcenterC + self.sagHR*self.normVectHR)
        self.set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)

    def _wedgeAngle_changed(self, old, new):
        self.set(trait_change_notify=False,
                 normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi))
        self.set(trait_change_notify=False,
                 normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle))
        self.set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)

    def _inv_ROC_HR_changed(self, old, new):
        #First update the sag
        if np.abs(self.inv_ROC_HR) > 1./(10*km):
            R = 1./self.inv_ROC_HR
            r = self.diameter/2
            self.sagHR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagHR = 0.0;
        #Update the HRcenterC
        self.set(trait_change_notify=False,
                 HRcenterC = self.HRcenter - self.sagHR*self.normVectHR)

    def _inv_ROC_AR_changed(self, old, new):
        #First update the sag
        if np.abs(self.inv_ROC_AR) > 1./(10*km):
            R = 1./self.inv_ROC_AR
            r = self.diameter/2
            self.sagAR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagAR = 0.0;

#}}}

#}}}

#{{{ Cylindrical Mirror Class

class CyMirror(Mirror):
    '''
    Representing a partial reflective cylindrical mirror. Note that both HR and AR surfaces are treated as cylindrical if you specify non-zero ROC for them. The curve  directions of the two surfaces must be the same.

    Attributes
    ----------
    curve_direction : str
        Either 'h' or 'v'. If it is 'h' the mirror is curved in horizontal plane. If 'v', it is vertical.
    HRcenter : array
        The position of the center of the arc of the HR surface. shape(2,).
    HRcenterC : array
        The position of the center of the chord of the HR surface. shape(2,).
    normVectHR : array
        Normal vector of the HR surface. shape(2,)
    normAngleHR : float
        Angle of the HR normal vector. In radians.
    ARcenter : array
        The position of the center of the AR surface. shape(2,)
    normVectAR : array
        Normal vector of the HR surface. shape(2,)
    normAngleAR : float
        Angle of the HR normal vector. In radians.
    HRtransmissive : boolean
        A boolean value defaults to False. If True, this mirror
        is supposed to transmit beams on the HR surface. Therefore,
        for the first encounter of a beam on the HR surface of this mirror
        will not increase the stray_order. This flag should be set to True for
        beam splitters and input test masses.
    term_on_HR : boolean
        If this is True, a beam with stray_order <= self.term_on_HR_order will be terminated when
        it hits on HR. This is to avoid the inifinite loop of non-sequencial
        trace by forming a cavity.
    term_on_HR_order : int
        Integer to specify the upper limit of the stray order used to judge
        whether to terminate the non sequential trace or not on HR reflection.
    '''

#{{{ __init__

    def __init__(self, HRcenter=[0.0,0.0], normAngleHR=0.0,
                 normVectHR=None, diameter=25.0*cm, thickness=15.0*cm,
                 wedgeAngle=0.25*pi/180., inv_ROC_HR=1.0/7000.0, inv_ROC_AR=0.0,
                 Refl_HR=0.99, Trans_HR=0.01, Refl_AR=0.01, Trans_AR=0.99, n=1.45,
                 name="Mirror", HRtransmissive=False, term_on_HR=False, curve_direction='h'):
        '''
        Create a cylindrical mirror object.

        Parameters
        ----------
        HRcenter : array, optional
            Position of the center of the HR surface.
            Defaults [0.0, 0.0].
        normAngleHR : float, optional
            Direction angle of the normal vector of the HR surface. In radians.
            Defaults 0.0.
        normVectHR : arrary or None, optional
            Normal vector of the HR surface. Should be an array of shape(2,).
            Defaults None.
        diameter : float, optional
            Diameter of the mirror.
            Defaults 25.0*cm.
        thickness : float, optional
            Thickness of the mirror.
            Defaults 15.0*cm.
        wedgeAngle : float, optional
            Wedge angle between the HR and AR surfaces. In radians.
            Defaults 0.25*pi/180.
        inv_ROC_HR : float, optional
            1/ROC of the HR surface.
            Defaults 1.0/7000.0.
        inv_ROC_AR : float, optional
            1/ROC of the AR surface.
            Defaults 0.0.
        Refl_HR : float, optional
            Power reflectivity of the HR surface.
            Defaults 0.99.
        Trans_HR : float, optional
            Power transmissivity of the HR surface.
            Defaults 0.01.
        Refl_AR : float, optional
            Power reflectivity of the AR surface.
            Defaults 0.01.
        Trans_AR : float, optional
            Power transmissivity of the AR surface.
            Defaults 0.99.
        n : float, optional
            Index of refraction.
            Defaults 1.45.
        name : str, optional
            Name of the mirror.
            Defaults "Mirror".
        HRtransmissive : boolean, optional
            If True, this mirror
            is supposed to transmit beams on the HR surface. Therefore,
            for the first encounter of a beam on the HR surface of this mirror
            will not increase the stray_order. This flag should be set to True for
            beam splitters and input test masses.
            Defaults False
        term_on_HR : boolean, optional
            If this is True, a beam with stray_order <= self.term_on_HR_order
            will be terminated when
            it hits on HR. This is to avoid the inifinite loop of
            non-sequencial
            trace by forming a cavity.
            Defaults False.
        curve_direction: str, optional
            Direction of curvature. Choose from ['h', 'v'].
            Defaults 'h'.
        '''
        self.diameter = diameter

        #Compute the sag.
        #Sag is positive for convex mirror.
        if np.abs(inv_ROC_HR) > 1./(10*km):
            R = 1./inv_ROC_HR
            r = self.diameter/2
            self.sagHR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagHR = 0.0;

        #Convert rotationAngle to normVectHR or vice versa.
        if normVectHR is not None:
            self.normVectHR = normVectHR
        else:
            self.normAngleHR = normAngleHR

        self.HRcenter = HRcenter
        self._HRcenter_changed(0,0)

        self.thickness = thickness
        self.wedgeAngle = wedgeAngle
        self.ARdiameter = self.diameter/np.cos(self.wedgeAngle)
        self.inv_ROC_HR = inv_ROC_HR
        self.inv_ROC_AR = inv_ROC_AR
        self.Refl_HR = Refl_HR
        self.Trans_HR = Trans_HR
        self.Refl_AR = Refl_AR
        self.Trans_AR = Trans_AR
        self.n = n
        self._normAngleHR_changed(0,0)
        self.name = name
        self.HRtransmissive = HRtransmissive
        self.term_on_HR = term_on_HR
        self.term_on_HR_order = 0
        self.curve_direction = curve_direction

#}}}

#{{{ copy

    def copy(self):
        return CyMirror(HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                      diameter=self.diameter, thickness=self.thickness,
                      wedgeAngle=self.wedgeAngle, inv_ROC_HR=self.inv_ROC_HR,
                      inv_ROC_AR=self.inv_ROC_AR, Refl_HR=self.Refl_HR,
                      Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR, Trans_AR=self.Trans_AR,
                      n=self.n, name=self.name, HRtransmissive=self.HRtransmissive,
                      term_on_HR=self.term_on_HR, curve_direction=self.curve_direction)

#}}}

#{{{ get_side_info

    def get_side_info(self):
        '''
        Return information on the sides of the mirror.
        Returned value is a list of two tuples like [(center1, normVect1, length1), (center2, normVect2, length2)]
        Each tuple corresponds to a side. center1 is the coordinates of the center of the side line. normVect1 is the normal vector of the side line. length1 is the length of the side line.

        Returns
        -------
        [(float, float, float), (float, float, float)]
        '''

        if self.curve_direction == 'v':
            center_of_HR =self.HRcenter
            thickness = self.thickness + self.sagHR + self.sagAR
        else:
            center_of_HR =self.HRcenterC
            thickness = self.thickness

        plVect = optics.geometric.vector_rotation_2D(self.normVectHR, pi/2)
        p1 = center_of_HR + plVect * self.diameter/2
        p2 = p1 - plVect * self.diameter
        p3 = p2 - self.normVectHR * (thickness - np.tan(self.wedgeAngle)*self.diameter/2)
        p4 = p1 - self.normVectHR * (thickness + np.tan(self.wedgeAngle)*self.diameter/2)


        center1 = (p1+p4)/2
        vn1 = optics.geometric.vector_rotation_2D(p1 - p4, pi/2)
        normVect1 = vn1/np.linalg.norm(vn1)
        length1 = np.linalg.norm(p1 - p4)

        center2 = (p2+p3)/2
        vn2 = optics.geometric.vector_rotation_2D(p2 - p3, -pi/2)
        normVect2 = vn2/np.linalg.norm(vn2)
        length2 = np.linalg.norm(p2 - p3)


        return [(center1, normVect1, length1), (center2, normVect2, length2)]



#}}}

#{{{ isHit

    def isHit(self, beam):
        '''
        A function to see if a beam hits this optics or not.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.

        Returns
        -------
        Dict
            The return value is a dictionary with the following keys:
            ``isHit, position, distance, face``

            ``isHit``:
            This is a boolean to answer whether the beam hit the optics
            or not.

            ``position``:
            A numpy array containing the coordinate values of the intersection
            point between the beam and the optics. If isHit is False, this parameter
            does not mean anything.

            ``distance``
            The distance between the beam origin and the intersection point.

            ``face``:
            An optional string identifying which face of the optics was hit.
            For example, ``face`` can be either "HR" or "AR" for a mirror.
            ``face`` can also be "side", meaning that the beam hits a side
            of the optics, which is not meant to be used, e.g. the side of a mirror.
            In this case, the beam have reached a dead end.
        '''

        if self.curve_direction == 'h':
            HRsurface = {'center': self.HRcenterC, 'normal_vector': self.normVectHR,
                         'size': self.diameter, 'inv_ROC': self.inv_ROC_HR, 'name': 'HR'}
            ARsurface = {'center': self.ARcenter, 'normal_vector': self.normVectAR,
                             'size': self.diameter, 'inv_ROC': self.inv_ROC_AR, 'name': 'AR'}
        else:
            HRsurface = {'center': self.HRcenter, 'normal_vector': self.normVectHR,
                         'size': self.diameter, 'inv_ROC': 0.0, 'name': 'HR'}
            ARsurface = {'center': self.ARcenter, 'normal_vector': self.normVectAR,
                             'size': self.diameter, 'inv_ROC': 0.0, 'name': 'AR'}

        # #The vector parallel to the HR surface, pointing left.
        # v1 = np.array((-self.normVectHR[1], self.normVectHR[0]))
        # #Left corner of the HR surface
        # c1 = self.HRcenterC + self.diameter/2 * v1
        # #Right corner of the HR surface
        # c2 = self.HRcenterC - self.diameter/2 * v1
        # #Center of the side 1
        # side_center_1 = c1 + self.thickness/2 * (- self.normVectHR)
        # #Center of the side 2
        # side_center_2 = c2 + self.thickness/2 * (- self.normVectHR)

        # Side2 = {'center': side_center_2, 'normal_vector': -v1,
        #              'size': self.thickness, 'inv_ROC': 0.0, 'name': 'side'}

        sides = self.get_side_info()

        Side1 = {'center': sides[0][0], 'normal_vector': sides[0][1],
                     'size': sides[0][2], 'inv_ROC': 0.0, 'name': 'side'}
        Side2 = {'center': sides[1][0], 'normal_vector': sides[1][1],
                     'size': sides[1][2], 'inv_ROC': 0.0, 'name': 'side'}

        faceList = [HRsurface, ARsurface, Side1, Side2]

        min_dist = 1e16
        final_answer = None
        for face in faceList:
            ans = self._isHitSurface_(beam, surface_center=face['center'],
                                normal_vector=face['normal_vector'],
                                surface_size=face['size'], inv_ROC=face['inv_ROC'])
            if ans['isHit']:
                if min_dist > ans['distance']:
                    min_dist = ans['distance']
                    final_answer = ans
                    face_name = face['name']

        if final_answer is None:
            return {'isHit': False, 'position': np.array((0,0)),
                'distance': 0.0, 'face':''}
        else:
            return {'isHit': True, 'position': final_answer['Intersection Point'],
                'distance': min_dist, 'face': face_name}



#}}}

#{{{ Draw

    def draw(self, cv, drawName=False):
        '''
        Draw itself
        '''

        if self.curve_direction == 'v':
            center_of_HR =self.HRcenter
            thickness = self.thickness + self.sagHR + self.sagAR
        else:
            center_of_HR =self.HRcenterC
            thickness = self.thickness

        plVect = optics.geometric.vector_rotation_2D(self.normVectHR, pi/2)
        p1 = center_of_HR + plVect * self.diameter/2
        p2 = p1 - plVect * self.diameter
        p3 = p2 - self.normVectHR * (thickness - np.tan(self.wedgeAngle)*self.diameter/2)
        p4 = p1 - self.normVectHR * (thickness + np.tan(self.wedgeAngle)*self.diameter/2)

        cv.add_shape(draw.Line(p2,p3), layername="Mirrors")
        cv.add_shape(draw.Line(p4,p1), layername="Mirrors")

        d = self.thickness/10
        l1 = p1 - self.normVectHR * d
        l2 = p2 - self.normVectHR * d
        cv.add_shape(draw.Line(l1,l2), layername="Mirrors")

        #Draw Curved surface

        #HR

        if np.abs(self.inv_ROC_HR) > 1.0/1e5 and self.curve_direction == 'h':
            R = 1/self.inv_ROC_HR
            theta = np.arcsin(self.diameter/2/R)
            sag = R*(1-np.cos(theta))
            x = np.linspace(0, self.diameter/2, 30)
            y = R*(1.0 - np.sqrt(1.0 - x**2/(R**2))) -sag
            x2 = -np.flipud(x)
            y2 = np.flipud(y)
            x = np.hstack((x2,x))
            y = np.hstack((y2,y))
            v = np.vstack((x,y))
            v = optics.geometric.vector_rotation_2D(v, self.normAngleHR - pi/2)
            v = v.T + self.HRcenterC
            cv.add_shape(draw.PolyLine(x=v[:,0], y=v[:,1]), layername="Mirrors")
            #dxf.append(sdxf.LwPolyLine(points=list(v), layer="Mirrors"))
        else:
            cv.add_shape(draw.Line(p1,p2), layername="Mirrors")
            #dxf.append(sdxf.Line(points=[p1,p2], layer="Mirrors"))

        #AR
        if np.abs(self.inv_ROC_AR) > 1.0/1e5 and self.curve_direction == 'h':
            diameter = self.diameter/np.cos(self.wedgeAngle)

            R = 1/self.inv_ROC_AR
            theta = np.arcsin(diameter/2/R)
            sag = R*(1-np.cos(theta))
            x = np.linspace(0, diameter/2, 30)
            y = R*(1.0 - np.sqrt(1.0 - x**2/(R**2))) -sag
            x2 = -np.flipud(x)
            y2 = np.flipud(y)
            x = np.hstack((x2,x))
            y = np.hstack((y2,y))
            v = np.vstack((x,y))
            v = optics.geometric.vector_rotation_2D(v, self.normAngleAR - pi/2)
            v = v.T + self.ARcenter
            cv.add_shape(draw.PolyLine(x=v[:,0], y=v[:,1]), layername="Mirrors")
            #dxf.append(sdxf.LwPolyLine(points=list(v), layer="Mirrors"))
        else:
            cv.add_shape(draw.Line(p3,p4), layername="Mirrors")
            #dxf.append(sdxf.Line(points=[p3,p4], layer="Mirrors"))


        if drawName:
            center = (p1+p2+p3+p4)/4.
            height = self.thickness/4.
            width = height*len(self.name)
            center = center - np.array([width/2, height/2])
            cv.add_shape(draw.Text(text=self.name, point=center,height=height),
                         layername="text")
            # dxf.append(sdxf.Text(text=self.name, point=center, #
            #                      height=height, layer='text'))


#}}}

#{{{ hitFromHR

    def hitFromHR(self, beam, order=0, threshold=0.0, verbose=False):
        '''
        Compute the reflected and deflected beams when
        an input beam hit the HR surface.

        The internal reflections are computed as long as the number
        of internal reflections are below the ``order`` and the power
        of the reflected beams is over the threshold.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.
        verbose : boolean, optional
            Print useful information.

        Returns
        -------
        beams : dict
            Dictionary of reflected and deflected beams.
        '''

        #A dictionary to hold beams
        beams={}

        if self.curve_direction == 'h':
            chord_center_HR = self.HRcenterC
            chord_center_AR = self.ARcenterC
            inv_ROC_HR = self.inv_ROC_HR
            inv_ROC_AR = self.inv_ROC_AR
        else:
            chord_center_HR = self.HRcenter
            chord_center_AR = self.ARcenter
            inv_ROC_HR = 0.0
            inv_ROC_AR = 0.0

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=chord_center_HR,
                                                     chordNormVect=self.normVectHR,
                                                     invROC=inv_ROC_HR,
                                                     diameter=self.diameter)
        if not ans['isHit']:
            #The input beam does not hit the mirror.
            if verbose:
                print((self.name + ': The beam does not hit the mirror'))
            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_in = beam.copy() #Make a copy
        beam_in.length = ans['distance']
        beam_in.incSurfAngle = localNormAngle
        beam_in.incSurfInvROC = inv_ROC_HR
        beams['input']= beam_in


        #Propagate the input beam to the intersection point
        beam_on_HR = beam_in.copy()
        beam_on_HR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                               localNormAngle,
                                               1.0, self.n, invROC=inv_ROC_HR)
        #Reflected beam
        beam_r1 = beam_on_HR.copy()
        beam_r1.P = beam_r1.P * self.Refl_HR
        if beam_r1.P > threshold:
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = inv_ROC_HR
            beam_r1.incSurfAngle = None
            beam_r1.incSurfInvROC = None
            beam_r1.name = self.name+':r1'
            beams['r1'] = beam_r1

        #Transmitted beam
        beam_s1 = beam_on_HR.copy()
        beam_s1.P = beam_s1.P * self.Trans_HR
        if not self.HRtransmissive:
            beam_s1.stray_order = beam_s1.stray_order+1
        if beam_s1.P < threshold or beam_s1.stray_order > order:
            return beams
        beam_s1.dirAngle = deflAngle
        beam_s1.n = self.n
        beam_s1.ABCDTrans(Mtx, Mty)
        beam_s1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
        beam_s1.departSurfInvROC = -inv_ROC_HR

        #Hit AR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=chord_center_AR,
                                                     chordNormVect=-self.normVectAR,
                                                     invROC=-inv_ROC_AR,
                                                     diameter=self.ARdiameter)

        if not ans['isHit']:
            #The beam does not hit the AR surface. It must hit either of the sides.

            #Get side information
            sides = self.get_side_info()

            #Loop for sides
            for side in sides:
                ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                if ans['isHit']:
                    localNormAngle = ans['localNormAngle']
                    beam_s1.length = ans['distance']
                    beam_s1.layer = 'aux_beam'
                    beam_s1.incSurfAngle = localNormAngle
                    beam_s1.incSurfInvROC = 0.0
                    beam_s1.name = self.name+':s1'
                    beams['s1']= beam_s1
                    return beams

            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_s1.length = ans['distance']
        beam_s1.incSurfAngle = localNormAngle
        beam_s1.incSurfInvROC = -inv_ROC_AR
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the AR surface
        beam_on_AR = beam_s1.copy()
        beam_on_AR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                               localNormAngle,
                                               self.n, 1.0, invROC=-inv_ROC_AR)

        #Transmitted beam
        beam_t1 = beam_on_AR.copy()
        beam_t1.P = beam_on_AR.P * self.Trans_AR
        if beam_t1.P > threshold:
            beam_t1.dirAngle = deflAngle
            beam_t1.n = 1.0
            beam_t1.ABCDTrans(Mtx, Mty)
            beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
            beam_t1.departSurfInvROC = inv_ROC_AR
            beam_t1.incSurfAngle = None
            beam_t1.incSurfInvROC = None
            beam_t1.name = self.name+':t%d'%(1)
            beams['t1'] = beam_t1


        #Reflected beam
        beam_sr = beam_on_AR.copy()
        beam_sr.P = beam_sr.P * self.Refl_AR
        beam_sr.stray_order = beam_sr.stray_order+1
        if beam_sr.P < threshold or beam_sr.stray_order > order:
            return beams
        beam_sr.dirAngle = reflAngle
        beam_sr.ABCDTrans(Mrx, Mry)
        beam_sr.departSurfAngle = localNormAngle
        beam_sr.departSurfInvROC = -inv_ROC_AR


        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit the HR from the back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=chord_center_HR,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-inv_ROC_HR,
                                                         diameter=self.diameter)


            if not ans['isHit']:
                #The beam does not hit the HR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_sr.length = ans['distance']
                        beam_sr.layer = 'aux_beam'
                        beam_sr.incSurfAngle = localNormAngle
                        beam_sr.incSurfInvROC = 0.0
                        beam_sr.name = self.name+':s%d'%(2*ii)
                        beams['s'+str(2*ii)]= beam_sr
                        break

                break

            #Local normal angle
            localNormAngle = ans['localNormAngle']

            beam_sr.length = ans['distance']
            beam_sr.layer = 'aux_beam'
            beam_sr.incSurfAngle = localNormAngle
            beam_sr.incSurfInvROC = -inv_ROC_HR
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr

            #Propagate the input beam to the intersection point
            beam_on_HR = beam_sr.copy()
            beam_on_HR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-inv_ROC_HR)

            #Transmitted through HR
            beam_r1 = beam_on_HR.copy()
            beam_r1.P = beam_r1.P * self.Trans_HR
            beam_r1.stray_order = beam_r1.stray_order+1
            if beam_r1.P > threshold and beam_r1.stray_order <= order:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = inv_ROC_HR
                beam_r1.incSurfAngle = None
                beam_r1.incSurfInvROC = None
                beam_r1.name = self.name+':r%d'%(ii+1)
                beams['r'+str(ii+1)] = beam_r1

            #Reflected by HR
            beam_s1 = beam_on_HR.copy()
            beam_s1.P = beam_s1.P * self.Refl_HR
            if beam_s1.P < threshold:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.P = beam_s1.P * self.Refl_HR
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -inv_ROC_HR

            #Hit AR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=chord_center_AR,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-inv_ROC_AR,
                                                         diameter=self.ARdiameter)

            if not ans['isHit']:
                #The beam does not hit the AR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_s1.length = ans['distance']
                        beam_s1.layer = 'aux_beam'
                        beam_s1.incSurfAngle = localNormAngle
                        beam_s1.incSurfInvROC = 0.0
                        beam_s1.name = self.name+':s%d'%(2*ii+1)
                        beams['s'+str(2*ii+1)]= beam_s1
                        break

                break


            #Local normal angle
            localNormAngle = ans['localNormAngle']

            beam_s1.incSurfAngle = localNormAngle
            beam_s1.incSurfInvROC = -inv_ROC_AR
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1

            #Propagate the beam to the AR surface
            beam_on_AR = beam_s1.copy()
            beam_on_AR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-inv_ROC_AR)
            #Transmitted beam
            beam_t1 = beam_on_AR.copy()
            beam_t1.P = beam_on_AR.P * self.Trans_AR
            if beam_t1.P > threshold:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = inv_ROC_AR
                beam_t1.incSurfAngle = None
                beam_t1.incSurfInvROC = None
                beam_t1.name = self.name+':t%d'%(ii+1)
                beams['t'+str(ii+1)] = beam_t1

            #Reflected beam
            beam_sr = beam_on_AR.copy()
            beam_sr.P = beam_sr.P * self.Refl_AR
            beam_sr.stray_order = beam_sr.stray_order+1
            if beam_sr.P < threshold or beam_sr.stray_order > order:
                break
            beam_sr.dirAngle = reflAngle
            beam_sr.ABCDTrans(Mrx, Mry)
            beam_sr.departSurfAngle = localNormAngle
            beam_sr.departSurfInvROC = -inv_ROC_AR

            ii=ii+1

        return beams
#}}}

#{{{ hitFromAR

    def hitFromAR(self, beam, order=0, threshold=0.0, verbose=False):
        '''
        Compute the reflected and deflected beams when
        an input beam hit the AR surface.

        The internal reflections are computed as long as the number
        of internal reflections are below the ``order`` and the power
        of the reflected beams is over the threshold.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            An integer to specify how many times the internal reflections
            are computed.
            Defaults 0.
        threshold : float, optional
            The power threshold for internal reflection calculation.
            If the power of an auxiliary beam falls below this threshold,
            further propagation of this beam will not be performed.
            Defaults 0.0.
        verbose : boolean, optional
            Print useful information.

        Returns
        -------
        beams : dict
            Dictionary of reflected and deflected beams.
        '''

        #A dictionary to hold beams
        beams={}

        if self.curve_direction == 'h':
            chord_center_HR = self.HRcenterC
            chord_center_AR = self.ARcenterC
            inv_ROC_HR = self.inv_ROC_HR
            inv_ROC_AR = self.inv_ROC_AR
        else:
            chord_center_HR = self.HRcenter
            chord_center_AR = self.ARcenter
            inv_ROC_HR = 0.0
            inv_ROC_AR = 0.0

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=chord_center_AR,
                                                     chordNormVect=self.normVectAR,
                                                     invROC=inv_ROC_AR,
                                                     diameter=self.ARdiameter)

        if not ans['isHit']:
            #The input beam does not hit the mirror.
            if verbose:
                print((self.name + ': The beam does not hit the mirror'))
            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']

        beam_in = beam.copy() #Make a copy
        beam_in.incSurfAngle = localNormAngle
        beam_in.incSurfInvROC = inv_ROC_AR
        beam_in.length = ans['distance']
        beams['input']= beam_in

        #Propagate the input beam to the intersection point
        beam_on_AR = beam_in.copy()
        beam_on_AR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                               localNormAngle,
                                               1.0, self.n, invROC=inv_ROC_AR)
        #Reflected beam
        beam_r1 = beam_on_AR.copy()
        beam_r1.P = beam_r1.P * self.Refl_AR
        beam_r1.stray_order = beam_r1.stray_order+1
        if beam_r1.P > threshold and beam_r1.stray_order <= order:
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = inv_ROC_AR
            beam_r1.incSurfAngle = None
            beam_r1.incSurfInvROC = None
            beam_r1.name = self.name+':r1'
            beams['r1'] = beam_r1

        #Transmitted beam
        beam_s1 = beam_on_AR.copy()
        beam_s1.P = beam_s1.P * self.Trans_AR
        if beam_s1.P < threshold:
            return beams
        beam_s1.dirAngle = deflAngle
        beam_s1.n = self.n
        beam_s1.ABCDTrans(Mtx, Mty)
        beam_s1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
        beam_s1.departSurfInvROC = -inv_ROC_AR

        #Hit HR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=chord_center_HR,
                                                     chordNormVect=-self.normVectHR,
                                                     invROC=-inv_ROC_HR,
                                                     diameter=self.diameter)

        if not ans['isHit']:
            #The beam does not hit the HR surface. It must hit either of the sides.

            #Get side information
            sides = self.get_side_info()

            #Loop for sides
            for side in sides:
                ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                if ans['isHit']:
                    localNormAngle = ans['localNormAngle']
                    beam_s1.length = ans['distance']
                    beam_s1.layer = 'aux_beam'
                    beam_s1.incSurfAngle = localNormAngle
                    beam_s1.incSurfInvROC = 0.0
                    beam_s1.name = self.name+':s1'
                    beams['s1']= beam_s1
                    return beams

            return beams

        #Local normal angle
        localNormAngle = ans['localNormAngle']
        beam_s1.length = ans['distance']
        beam_s1.incSurfAngle = localNormAngle
        beam_s1.incSurfInvROC = -inv_ROC_HR
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the HR surface
        beam_on_HR = beam_s1.copy()
        beam_on_HR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                               localNormAngle,
                                               self.n, 1.0, invROC=-inv_ROC_HR)

        #Transmitted beam
        beam_t1 = beam_on_HR.copy()
        beam_t1.P = beam_on_HR.P * self.Trans_HR
        if not self.HRtransmissive:
            beam_t1.stray_order = beam_t1.stray_order+1
        if beam_t1.P > threshold and beam_t1.stray_order <= order:
            beam_t1.dirAngle = deflAngle
            beam_t1.n = 1.0
            beam_t1.ABCDTrans(Mtx, Mty)
            beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
            beam_t1.departSurfInvROC = inv_ROC_HR
            beam_t1.incSurfAngle = None
            beam_t1.incSurfInvROC = None
            beam_t1.name = self.name+':t1'
            beams['t1'] = beam_t1

        #Reflected beam
        beam_sr = beam_on_HR.copy()
        beam_sr.P = beam_sr.P * self.Refl_HR
        if beam_sr.P < threshold:
            return beams
        beam_sr.dirAngle = reflAngle
        beam_sr.ABCDTrans(Mrx, Mry)
        beam_sr.departSurfAngle = localNormAngle
        beam_sr.departSurfInvROC = -inv_ROC_HR

        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit AR from back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=chord_center_AR,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-inv_ROC_AR,
                                                         diameter=self.ARdiameter)

            if not ans['isHit']:
                #The beam does not hit the AR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_sr.length = ans['distance']
                        beam_sr.layer = 'aux_beam'
                        beam_sr.incSurfAngle = localNormAngle
                        beam_sr.incSurfInvROC = 0.0
                        beam_sr.name = self.name+':s%d'%(2*ii)
                        beams['s'+str(2*ii)]= beam_sr
                        break

                break

            #Local normal angle
            localNormAngle = ans['localNormAngle']
            beam_sr.length = ans['distance']
            beam_sr.layer = 'aux_beam'
            beam_sr.incSurfAngle = localNormAngle
            beam_sr.incSurfInvROC = -inv_ROC_AR
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr


            #Propagate the input beam to the intersection point
            beam_on_AR = beam_sr.copy()
            beam_on_AR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_AR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-inv_ROC_AR)

            #Transmitted through AR
            beam_r1 = beam_on_AR.copy()
            beam_r1.P = beam_r1.P * self.Trans_AR
            if beam_r1.P > threshold:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = inv_ROC_AR
                beam_r1.incSurfAngle = None
                beam_r1.incSurfInvROC = None
                beam_r1.name = self.name+':r%d'%(ii+1)
                beams['r'+str(ii+1)] = beam_r1

            #Reflected by AR
            beam_s1 = beam_on_AR.copy()
            beam_s1.P = beam_s1.P * self.Refl_AR
            beam_s1.stray_order = beam_s1.stray_order+1
            if beam_s1.P < threshold or beam_s1.stray_order > order:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -inv_ROC_AR

            #Hit HR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=chord_center_HR,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-inv_ROC_HR,
                                                         diameter=self.diameter)

            if not ans['isHit']:
                #The beam does not hit the HR surface. It must hit either of the sides.

                #Get side information
                sides = self.get_side_info()

                #Loop for sides
                for side in sides:
                    ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                                 chord_center=side[0],
                                                                 chordNormVect=-side[1],
                                                                 invROC=0.0,
                                                                 diameter=side[2])
                    if ans['isHit']:
                        localNormAngle = ans['localNormAngle']
                        beam_s1.length = ans['distance']
                        beam_s1.layer = 'aux_beam'
                        beam_s1.incSurfAngle = localNormAngle
                        beam_s1.incSurfInvROC = 0.0
                        beam_s1.name = self.name+':s%d'%(2*ii+1)
                        beams['s'+str(2*ii+1)]= beam_s1
                        break

                break

           #Local normal angle
            localNormAngle = ans['localNormAngle']
            beam_s1.incSurfAngle = localNormAngle
            beam_s1.incSurfInvROC = -inv_ROC_HR
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1


            #Propagate the beam to the HR surface
            beam_on_HR = beam_s1.copy()
            beam_on_HR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.refl_defl_angle(beam_on_HR.dirAngle,
                                                         localNormAngle,
                                                         self.n, 1.0, invROC=-inv_ROC_HR)

            #Transmitted beam
            beam_t1 = beam_on_HR.copy()
            beam_t1.P = beam_t1.P * self.Trans_HR
            beam_t1.stray_order = beam_t1.stray_order+1
            if beam_t1.P > threshold and beam_t1.stray_order <= order:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = inv_ROC_HR
                beam_t1.incSurfAngle = None
                beam_t1.incSurfInvROC = None
                beam_t1.name = self.name+':t%d'%(ii+1)
                beams['t'+str(ii+1)] = beam_t1

            #Reflected beam
            beam_sr = beam_on_HR.copy()
            beam_sr.P = beam_sr.P * self.Refl_HR
            if beam_sr.P < threshold:
                break
            beam_sr.dirAngle = reflAngle
            beam_sr.ABCDTrans(Mrx, Mry)
            beam_sr.departSurfAngle = localNormAngle
            beam_sr.departSurfInvROC = -inv_ROC_HR

            ii=ii+1

        return beams

#}}}

#}}}
