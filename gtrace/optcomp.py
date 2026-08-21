'''
Define optical components for gtrace.

'''
#{{{ Import modules
import numpy as np
pi = np.pi
array = np.array
sqrt = np.lib.scimath.sqrt
from numpy.linalg import norm

from traits.api import (Any, HasTraits, Enum, Int, Float, CFloat, CArray,
                        List, Str, Union)

import gtrace.optics as optics
import gtrace.optics.geometric
from .unit import *
import copy
import math
import gtrace.draw as draw

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
__version__ = "0.6.0"
__maintainer__ = "Yoichi Aso"
__email__ = "asoy01@gmail.com"
__status__ = "Beta"

#}}}

#{{{ sagitta

def _sagitta(invROC, r):
    '''
    How far the middle of an arc stands off its own chord.

    Parameters
    ----------
    invROC : float
        Inverse radius of curvature. Zero is a flat face, which stands
        off nothing.
    r : float
        Half the chord length.

    Returns
    -------
    float
        Never negative. The answer is a distance, and which side of the
        chord the arc is on is not asked here.
    '''
    if invROC == 0.0:
        return 0.0
    Rc = abs(1.0/invROC)
    if r >= Rc:
        #A hemisphere or more. The apex is a full radius off the chord,
        #and the square root below would be of a negative number.
        return Rc
    return Rc - math.sqrt(Rc*Rc - r*r)

#}}}

#{{{ Probe ray

class _ProbeRay(object):
    '''
    A bare ray to ask an optics where its surfaces are.

    isHit() reads nothing but the origin and the direction of what it is
    given, so asking it a question about geometry does not need a beam
    with a q parameter, a power and a wavelength. Handing it a
    GaussianBeam would mean inventing those, and they would be a
    fiction: none of them takes part in the answer. verify_surfaces.py
    holds isHit() to that, by asking it the same question both ways.
    '''
    def __init__(self, pos, dirVect):
        self.pos = np.array(pos, dtype='float64')
        self.dirVect = np.array(dirVect, dtype='float64')

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
    max_stray_order : int or None
        Upper limit of the stray order computed when a beam hits this
        optics, overriding the order given to non_seq_trace. None means
        the trace-wide order is used.

        How deep the ghost beams inside a substrate are worth chasing
        depends on the part an element plays in the system - the ghosts
        of a beam splitter usually matter, those of a steering mirror
        usually do not - so it belongs to the element, next to
        term_on_HR, rather than to the trace.
    '''
    name = Str()
    center = CArray(dtype='float64', shape=(2,))
    rotationAngle = CFloat(0.0) #in rad
    max_stray_order = Union(None, Int)

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
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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

            ``face``
            An optional string identifying which face of the optics was hit.
            For a mirror, ``face`` is any of "HR", "AR" or "side".
        '''
        #This is an abstract function
        return {False, {}, "side"}  # Is this a bug? Shouldn't it be a tuple?

#}}}

#{{{ get_corners(), contains_segment()

    def get_corners(self):
        '''
        The corners of the substrate, going around it.

        Where get_side_info() describes the two sides as centre, normal
        and length - which is what hit testing needs - this gives the
        points themselves, which is what anything pointing at the
        element needs: a front end snapping a measurement to a corner,
        or a drawing that wants the outline.

        The corners sit on the chord planes of the two faces, not on the
        faces themselves: a curved face leaves its apex a sagitta away
        from the line joining its two corners.

        Returns
        -------
        list of numpy.ndarray
            Four points of shape (2,), starting at one end of the front
            face and going round the substrate.
        '''
        #This is an abstract function
        return []

    def contains_segment(self, p1, p2, rtol=1e-9):
        '''
        Whether the straight segment from p1 to p2 lies wholly inside
        this substrate.

        A point is measured against the same surfaces the tracer hits,
        so the answer agrees with what a beam would do. isHit() reports
        a surface only when it is approached from outside - it rejects
        any face the ray is leaving through - and that is what makes
        this simple: from inside the substrate, isHit() finds nothing at
        all, and an entry found partway along the segment means the
        segment starts outside.

        Endpoints exactly on a face count as inside, since that is where
        a measurement is usually taken from: the optical thickness of a
        substrate is measured between the apexes of its two faces.

        Parameters
        ----------
        p1, p2 : array_like
            The ends of the segment, in global coordinates.
        rtol : float, optional
            How close to an end an entry may be found and still be read
            as that end sitting on the face, relative to the length of
            the segment. Defaults 1e-9.

        Returns
        -------
        bool
        '''
        p1 = np.array(p1, dtype='float64')
        p2 = np.array(p2, dtype='float64')
        seg = p2 - p1
        L = np.linalg.norm(seg)
        if L == 0.0:
            return False
        dirVect = seg / L
        tol = rtol * L

        #An entry found between the ends means the segment crosses into
        #the substrate rather than running inside it. Both ends are
        #asked, because isHit() only ever looks forward, and an entry
        #right at an end is the end sitting on the face.
        for pos, d in ((p1, dirVect), (p2, -dirVect)):
            ans = self.isHit(_ProbeRay(pos, d))
            if ans['isHit'] and tol < ans['distance'] < L - tol:
                return False

        #No crossing in between leaves two cases apart: the segment runs
        #inside the substrate, or it misses it entirely. One interior
        #point settles it, and the midpoint is as good as any: a segment
        #that wandered out of the substrate and back would have to come
        #back in through a face, and the entry would have been found
        #above.
        #
        #The point is not asked along the segment: a segment drawn
        #corner to corner runs along the chord of a face, where the line
        #meets the sides exactly at their ends and the answer turns on
        #which way a tangent rounds. Asking along the front face normal
        #instead puts the question where the faces are square to it.
        return self.contains_point(p1 + seg / 2, rtol=rtol)

    def contains_point(self, point, dirVect=None, rtol=1e-9):
        '''
        Whether a point lies inside this substrate.

        Three questions, all put to isHit(), which reports a surface
        only when it is approached from outside:

        - looking one way from the point, is a surface entered?
        - looking the other way, is one entered?
        - coming from well outside, does the substrate begin before the
          point is reached?

        From inside the substrate the first two find nothing: every face
        is one the ray is leaving through. Either of them finding
        something places the point outside - including the awkward case
        of a point in the hollow of a concave face, which is enclosed by
        the substrate on three sides and would fool any test that only
        asked how far the material reaches. The third question is what
        separates a point inside from one out in the open, where nothing
        is entered either.

        A point on a face counts as inside, to within rtol of the reach
        of the probes.

        Parameters
        ----------
        point : array_like
            The point, in global coordinates.
        dirVect : array_like, optional
            The line to look along. Defaults to the front face normal.
        rtol : float, optional
            Tolerance, relative to the reach of the probes. Defaults
            1e-9.

        Returns
        -------
        bool
        '''
        point = np.array(point, dtype='float64')
        if dirVect is None:
            dirVect = np.array(self.normVectHR, dtype='float64')
        else:
            dirVect = np.array(dirVect, dtype='float64')
            dirVect = dirVect / np.linalg.norm(dirVect)

        reach = self._probe_reach(point)
        tol = rtol * reach
        for e in (dirVect, -dirVect):
            ans = self.isHit(_ProbeRay(point, e))
            if ans['isHit'] and ans['distance'] > tol:
                return False
        ahead = self.isHit(_ProbeRay(point - dirVect * reach, dirVect))
        return bool(ahead['isHit']) and ahead['distance'] <= reach + tol

    def _probe_reach(self, point):
        '''
        A distance that starts a probe ray outside this substrate,
        whatever direction it comes from.
        '''
        centre = np.array(self.center, dtype='float64')
        radius = np.hypot(getattr(self, 'diameter', 0.0) / 2,
                          getattr(self, 'thickness', 0.0) / 2)
        return 2 * radius + np.linalg.norm(np.array(point,
                                                    dtype='float64') - centre)

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
            "localNormVect": A numpy array representing the normal vector
            of the surface at the intersection point.
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
        The position of the apex of the arc of the AR surface. shape(2,).
        Note that this is the counterpart of HRcenter, not of HRcenterC:
        it lies one sagitta out of the substrate. Anything that asks for
        the centre of a chord - line_arc_intersection(), or an arc drawn
        from its own chord - wants ARcenterC instead.
    ARcenterC : array
        The position of the center of the chord of the AR surface.
        shape(2,).
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
    HRreflective : boolean
        A boolean value defaults to True. If False, the HR surface of
        this optics is not supposed to reflect beams. Therefore, every
        reflection at the HR surface, from outside or from inside the
        substrate, increases the stray_order: such a reflection is a
        ghost. The mirror image of HRtransmissive. This flag should be
        set to False for lenses.
    term_on_HR : boolean
        If this is True, a beam with stray_order <= self.term_on_HR_order will be terminated when
        it hits on HR. This is to avoid the inifinite loop of non-sequencial
        trace by forming a cavity.
    term_on_HR_order : int
        Integer to specify the upper limit of the stray order used to judge
        whether to terminate the non sequential trace or not on HR reflection.
    term_on_HR_transmits : boolean
        What term_on_HR stops. False, the default, stops the beam at the
        surface: nothing is computed, which is what term_on_HR has always
        done. True stops only the reflection that would form the cavity,
        and lets the element be hit as usual otherwise, so the beam
        transmitted through the substrate carries on and the ghosts
        inside it are unfolded, counted and capped by order and
        max_stray_order like any others. Only the external reflection
        off the HR is dropped; a ghost leaving through the HR from
        inside the substrate is not. Has no effect unless term_on_HR is
        True, as term_on_HR_order has none.
    anchor_point : str
        The point the optics is held by: 'HRcenter', the apex of the HR
        arc, or 'center', the middle of the substrate. It is the point
        that stays put when inv_ROC_HR changes - the other one moves,
        since the sagitta between them is what the curvature changed -
        and the point the optics turns about when normAngleHR or
        normVectHR is assigned.

        Defaults to 'HRcenter'. Regrinding a telescope mirror is done to
        change the magnification, not to move the beam, and a layout
        puts the spot on the HR surface, so the arc has to stay under
        the spot while the substrate moves back behind it; steering a
        mirror likewise pivots the reflection point, not the substrate.

        'center' is for an optics the beam goes through rather than off,
        where the substrate is what is bolted to the bench and the faces
        are free to move on it. Lens defaults to it.
    '''

#{{{ Traits definitions

    HRcenter = CArray(dtype='float64', shape=(2,))
    HRcenterC = CArray(dtype='float64', shape=(2,))
    sagHR = CFloat()
    normVectHR = CArray(dtype='float64', shape=(2,))
    normAngleHR = CFloat()

    ARcenter = CArray(dtype='float64', shape=(2,))
    ARcenterC = CArray(dtype='float64', shape=(2,))
    sagAR = CFloat()
    normVectAR = CArray(dtype='float64', shape=(2,))
    normAngleAR = CFloat()

    diameter = CFloat(25.0*cm) #
    ARdiameter = CFloat()
    thickness = CFloat(15.0*cm) #
    wedgeAngle = CFloat(0.25*pi/180) # in rad
    n = CFloat(1.45) #Index of refraction

    inv_ROC_HR = CFloat(1.0/7000.0) #Inverse of the ROC of the HR surface.
    inv_ROC_AR = CFloat(0.0) #Inverse of the ROC of the AR surface.

    #Quantities that follow from the shape and the pose alone, kept
    #alongside the pose they were computed for. See _geometry_key.
    #transient, so that it is not written into a pickle: it is a
    #restatement of the traits above and would only be one more thing
    #that could disagree with them.
    _geom_cache = Any(transient=True)

    Refl_HR = CFloat(99.0) #Power reflectivity of the HR side.
    Trans_HR = CFloat(1.0) #Power transmittance of the HR side.

    Refl_AR = CFloat(0.01) #Power reflectivity of the AR side.
    Trans_AR = CFloat(99.99) #Power transmittance of the HR side.

    #Which point of the substrate stays put when the HR curvature
    #changes. See the class docstring.
    anchor_point = Enum(['HRcenter', 'center'])

    #Whether draw() marks the reflective side with a line just inside
    #the HR face. It says which face carries the coating, which is worth
    #saying for a mirror and misleading for a substrate meant to
    #transmit, so Lens turns it off.
    draw_HR_marker = True

    #What this element follows, if anything: the other element or the
    #body it is assembled to, where its anchor point sits in that
    #host's frame, how far it is turned relative to it, and whether
    #that relative angle is frozen. Two faces of a beam dump in a V
    #are one assembly, and so is a periscope.
    #
    #Plain attributes rather than traits, and never read by the optics
    #itself: the pose written here is the element's own, and an
    #assembly is a relation between two registered things, so it is
    #OpticalLayout that makes and keeps it - see assemble() and
    #_settle_assemblies(). They are declared here so that an element
    #can be asked what it follows without knowing whether it follows
    #anything.
    assembled_to = None
    assembly_offset = None
    assembly_angle = 0.0
    fix_rotation = True
    _assemble_name = None


#}}}

#{{{ __init__

    def __init__(self, HRcenter=[0.0,0.0], normAngleHR=0.0,
                 normVectHR=None, diameter=25.0*cm, thickness=15.0*cm,
                 wedgeAngle=0.25*pi/180., inv_ROC_HR=1.0/7000.0, inv_ROC_AR=0.0,
                 Refl_HR=0.99, Trans_HR=0.01, Refl_AR=0.01, Trans_AR=0.99, n=1.45,
                 name="Mirror", HRtransmissive=False, HRreflective=True,
                 term_on_HR=False, term_on_HR_order=0,
                 term_on_HR_transmits=False,
                 max_stray_order=None):
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
        HRreflective : boolean, optional
            If False, every reflection at the HR surface, from either
            side, increases the stray_order: this optics is not
            supposed to reflect there, so such a reflection is a ghost.
            Set to False for lenses.
            Defaults True
        term_on_HR : boolean, optional
            If this is True, a beam with stray_order <= self.term_on_HR_order
            will be terminated when
            it hits on HR. This is to avoid the inifinite loop of
            non-sequencial
            trace by forming a cavity.
            Defaults False.
        term_on_HR_order : int, optional
            Upper limit of the stray order at which term_on_HR still
            terminates a beam. A beam of a higher stray order is left
            alone.
            Defaults 0.
        term_on_HR_transmits : boolean, optional
            What term_on_HR stops. False stops the beam at the
            surface and computes nothing. True stops only the
            external reflection off the HR, so the beam transmitted
            through the substrate carries on. See the class
            docstring.
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
        self.HRreflective = HRreflective
        self.term_on_HR = term_on_HR
        self.term_on_HR_order = term_on_HR_order
        self.term_on_HR_transmits = term_on_HR_transmits
        self.max_stray_order = max_stray_order

#}}}

#{{{ copy

    def copy(self):
        m = Mirror(HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                      diameter=self.diameter, thickness=self.thickness,
                      wedgeAngle=self.wedgeAngle, inv_ROC_HR=self.inv_ROC_HR,
                      inv_ROC_AR=self.inv_ROC_AR, Refl_HR=self.Refl_HR,
                      Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR, Trans_AR=self.Trans_AR,
                      n=self.n, name=self.name, HRtransmissive=self.HRtransmissive,
                      HRreflective=self.HRreflective,
                      term_on_HR=self.term_on_HR,
                      term_on_HR_order=self.term_on_HR_order,
                      term_on_HR_transmits=self.term_on_HR_transmits,
                      max_stray_order=self.max_stray_order)
        #Not a constructor argument: it says what a later change to the
        #curvature does, and construction has none.
        m.anchor_point = self.anchor_point
        return m

#}}}

#{{{ cached geometry

    def _geometry_key(self):
        '''
        Every value the derived geometry below is computed from.

        Used to decide whether a cached result still describes this
        optics: it is recomputed when the key differs from the one it
        was computed for. Comparing the values is deliberate, rather
        than listening for trait changes. An in-place write such as
        ``opt.center[0] = 5`` fires no notification, so a cache
        invalidated by notification would go on returning the geometry
        of the old position, and nothing would say so.

        Subclasses that compute their geometry from anything else must
        extend this. CyMirror does, for curve_direction.

        Returns
        -------
        tuple
            Comparable by value, and cheap to build - well under a
            microsecond, against tens of microseconds to recompute what
            it guards.
        '''
        c = self.center
        h = self.HRcenterC
        v = self.normVectHR
        return (self.diameter, self.ARdiameter, self.thickness,
                self.wedgeAngle, self.sagHR, self.sagAR,
                self.inv_ROC_HR, self.inv_ROC_AR, self.normAngleHR,
                c[0], c[1], h[0], h[1], v[0], v[1])

    def _geometry(self):
        '''
        The store of derived geometry for the current shape and pose.

        Returns an empty dict once either has changed, so a caller finds
        what it put there only while that is still what the optics looks
        like.

        Returns
        -------
        dict
            Whatever the callers have put in it, keyed by name.
        '''
        key = self._geometry_key()
        cache = self._geom_cache
        if cache is None or cache[0] != key:
            cache = (key, {})
            self._geom_cache = cache
        return cache[1]

    def _bounding_radius(self):
        '''
        The radius of a circle about ``center`` that contains the whole
        substrate.

        isHit() uses it to reject a beam before testing the four faces
        one at a time. It must never come out too small: a radius that
        cuts inside the substrate makes a beam that does hit the optics
        be reported as missing, and that is a wrong answer with nothing
        to announce it. So the corners are taken exactly, and the bulge
        of a curved face is added on top of them rather than fitted in.

        Returns
        -------
        float
        '''
        geom = self._geometry()
        R = geom.get('bounding_radius')
        if R is not None:
            return R

        cx = self.center[0]
        cy = self.center[1]
        R = 0.0
        for corner in self.get_corners():
            d = math.hypot(corner[0] - cx, corner[1] - cy)
            if d > R:
                R = d

        #A face is drawn from the two ends of its chord, which is what
        #the corners are, and its arc stands off that chord by the
        #sagitta.
        R = R + max(_sagitta(self.inv_ROC_HR, self.diameter/2.0),
                    _sagitta(self.inv_ROC_AR, self.ARdiameter/2.0))

        geom['bounding_radius'] = R
        return R

    def _misses_bounding_circle(self, beam):
        '''
        Whether the beam stays clear of the circle that holds the whole
        substrate, and so cannot touch any face of it.

        A test isHit() puts in front of the per-face ones. It is allowed
        to say False about a beam that misses - the faces are then asked
        and give the right answer - but it must never say True about one
        that hits. Tracing the KAGRA interferometer, 95% of the calls
        end here.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam or _ProbeRay
            Anything with ``pos`` and ``dirVect``.

        Returns
        -------
        bool
        '''
        dx = beam.dirVect[0]
        dy = beam.dirVect[1]
        dlen = math.hypot(dx, dy)
        if dlen == 0.0:
            #No direction to reason about. Leave the answer to the faces.
            return False
        #A GaussianBeam keeps dirVect normalized, but _ProbeRay is given
        #whatever the caller wrote, so do not assume it.
        dx = dx/dlen
        dy = dy/dlen

        R = self._bounding_radius()
        ex = self.center[0] - beam.pos[0]
        ey = self.center[1] - beam.pos[1]

        #How far along the beam its closest approach to the centre lies.
        t = ex*dx + ey*dy
        if t < -R:
            #The whole substrate is behind where the beam starts.
            return True

        #How far off the beam the centre is at that point.
        ux = ex - t*dx
        uy = ey - t*dy
        return ux*ux + uy*uy > R*R

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

#{{{ get_corners

    def get_corners(self):
        '''
        The four corners of the substrate: the two ends of the HR chord
        first, then the two ends of the AR chord, going round.

        The wedge is in here - it is what makes the two sides different
        lengths - and so is the fact that a curved face meets the sides
        at its chord rather than at its apex.

        Returns
        -------
        list of numpy.ndarray
            Four points of shape (2,): HR left, HR right, AR right,
            AR left, where left and right are across the front face.
        '''
        plVect = optics.geometric.vector_rotation_2D(self.normVectHR, pi/2)
        p1 = self.HRcenterC + plVect * self.diameter/2
        p2 = p1 - plVect * self.diameter
        p3 = p2 - self.normVectHR * (self.thickness
                                     - np.tan(self.wedgeAngle)*self.diameter/2)
        p4 = p1 - self.normVectHR * (self.thickness
                                     + np.tan(self.wedgeAngle)*self.diameter/2)
        return [p1, p2, p3, p4]



#}}}

#{{{ rotate

    def rotate(self, angle, center=False):
        '''
        Rotate the optics rigidly about a pivot.

        Parameters
        ----------
        angle : float
            Angle of rotation, in radians.
        center : boolean or array, optional
            The pivot. False, the default, is whatever point
            ``anchor_point`` names - the point the optics is held by,
            and the pivot the viewer turns it about. A mirror anchors
            on the apex of its HR face, so for every mirror this is
            what rotate() has always done; a lens anchors on the middle
            of its substrate and turns about that. True is the middle
            of the substrate whatever the anchor says. An array is a
            point in global coordinates.
        '''
        if center is True:
            pivot = np.array(self.center)
        elif center is False:
            pivot = np.array(self.center if self.anchor_point == 'center'
                             else self.HRcenter)
        else:
            pivot = np.array(center)

        #The orientation first - the assignment turns the substrate
        #about its anchor point, wherever that is - and then HRcenter is
        #placed where a rigid turn about the pivot puts it, which fully
        #determines the position whatever the assignment did.
        h0 = np.array(self.HRcenter)
        self.normAngleHR = self.normAngleHR + angle
        self.HRcenter = pivot + optics.geometric.vector_rotation_2D(
            h0 - pivot, angle)
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

        if self.draw_HR_marker:
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
            #The polyline above is measured from the chord of the arc, so
            #it belongs on the chord centre. ARcenter is the apex, one
            #sagitta further out, and putting the arc there would leave
            #the AR surface hanging off the ends of the sides.
            v = v.T + self.ARcenterC
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

        #One cheap question before the four expensive ones. Most beams
        #in a layout of any size go nowhere near a given optics, and
        #each of those would otherwise be answered by intersecting four
        #faces and finding nothing.
        if self._misses_bounding_circle(beam):
            return {'isHit': False, 'position': np.array((0., 0.)),
                    'distance': 0.0, 'face': ''}

        HRsurface = {'center': self.HRcenterC, 'normal_vector': self.normVectHR,
                     'size': self.diameter, 'inv_ROC': self.inv_ROC_HR, 'name': 'HR'}
        #The AR surface has to be described here exactly as hitFromAR()
        #describes it: on its chord centre, with its own curvature and
        #its own chord length. Calling it flat made isHit() answer for a
        #different surface than the one the beam is then traced against,
        #which is invisible while the AR is flat and first order once it
        #is not.
        ARsurface = {'center': self.ARcenterC, 'normal_vector': self.normVectAR,
                     'size': self.ARdiameter, 'inv_ROC': self.inv_ROC_AR,
                     'name': 'AR'}

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
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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

            ``face``
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

        A beam is produced as long as its stray_order is not above
        ``order`` and its power is over the threshold. The stray_order
        of the incident beam is carried over, so a beam that is already
        stray leaves less of the allowance for the ghosts made here.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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
        #A face meant to reflect makes no ghost here: the reflection is
        #what the face is for, so the beam leaves at the stray order it
        #arrived at and `order` - which counts ghost generations - has
        #nothing to say about it. An already-stray beam reflects off a
        #mirror like any other. A face not meant to reflect does make a
        #ghost, and that one is counted and capped.
        ghost = not self.HRreflective
        if ghost:
            beam_r1.stray_order = beam_r1.stray_order+1
        if beam_r1.P > threshold and not (ghost
                                          and beam_r1.stray_order > order):
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
                                                     chord_center=self.ARcenterC,
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
            if not self.HRreflective:
                beam_s1.stray_order = beam_s1.stray_order+1
            if beam_s1.P < threshold or beam_s1.stray_order > order:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -self.inv_ROC_HR

            #Hit AR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=self.ARcenterC,
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

        A beam is produced as long as its stray_order is not above
        ``order`` and its power is over the threshold. The stray_order
        of the incident beam is carried over, so a beam that is already
        stray leaves less of the allowance for the ghosts made here.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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
                                                     chord_center=self.ARcenterC,
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
        if not self.HRreflective:
            beam_sr.stray_order = beam_sr.stray_order+1
        if beam_sr.P < threshold or beam_sr.stray_order > order:
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
                                                         chord_center=self.ARcenterC,
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
            if not self.HRreflective:
                beam_sr.stray_order = beam_sr.stray_order+1
            if beam_sr.P < threshold or beam_sr.stray_order > order:
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
        #The point the optics is held by must not move when it turns.
        #Captured before any geometry is rewritten, while the positions
        #still describe the old orientation. Only when the angle really
        #changed: __init__ calls this by hand with (0, 0) to settle the
        #derived geometry, at a moment when center is still computed
        #from half-assigned traits and must not be re-imposed.
        anchor = (np.array(self.center)
                  if self.anchor_point == 'center' and new != old
                  else None)

        self.trait_set(trait_change_notify=False,
                 normVectHR = array([np.cos(self.normAngleHR), np.sin(self.normAngleHR)]))
        self.trait_set(trait_change_notify=False,
                 normAngleHR = np.mod(self.normAngleHR, 2*pi))

        self.normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle)
        self.normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi)
        self.HRcenterC = self.HRcenter - self.normVectHR * self.sagHR

        #The update above pinned HRcenter, which is the anchor of every
        #mirror and costs nothing to keep. When the anchor is the middle
        #of the substrate, put it back: a rigid turn about any pivot
        #followed by the translation that returns the anchor is exactly
        #the turn about the anchor. _center_changed rebuilds every
        #position from it, all silently, so nothing comes back round.
        if anchor is not None:
            self.center = anchor

    def _normVectHR_changed(self, old, new):
        #See _normAngleHR_changed; assigning the vector is the same
        #turn ordered by other means.
        anchor = (np.array(self.center)
                  if self.anchor_point == 'center'
                  and not np.array_equal(old, new)
                  else None)

        #Normalize
        self.trait_set(trait_change_notify=False,
                 normVectHR = self.normVectHR/np.linalg.norm(array(self.normVectHR)))
        #Update dirAngle accordingly
        self.trait_set(trait_change_notify=False,
                 normAngleHR = np.mod(np.arctan2(self.normVectHR[1],
                                                   self.normVectHR[0]), 2*pi))

        self.normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle)
        self.normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi)
        self.HRcenterC = self.HRcenter - self.normVectHR * self.sagHR

        if anchor is not None:
            self.center = anchor

    def _HRcenterC_changed(self, old, new):
        self.trait_set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.trait_set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)
        #center is the middle of the substrate, so it lies between the
        #two chord planes. Averaging with ARcenter instead would mix a
        #chord centre with a point on the arc, putting center half a
        #sagitta out and disagreeing with _HRcenter_changed, with
        #_center_changed's inverse and with get_side_info.
        self.trait_set(trait_change_notify=False,
                 center = (self.HRcenterC + self.ARcenterC)/2.0)
        self.trait_set(trait_change_notify=False,
                 HRcenter = self.HRcenterC + self.sagHR*self.normVectHR)

    def _HRcenter_changed(self, old, new):
        self.trait_set(trait_change_notify=False,
                 HRcenterC = self.HRcenter - self.sagHR*self.normVectHR)
        self.trait_set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.trait_set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)
        self.trait_set(trait_change_notify=False,
                 center = (self.HRcenterC + self.ARcenterC)/2.0)

    def _center_changed(self, old, new):
        self.trait_set(trait_change_notify=False,
                 HRcenterC = self.center + self.normVectHR * self.thickness/2.0)
        self.trait_set(trait_change_notify=False,
                 HRcenter = self.HRcenterC + self.sagHR*self.normVectHR)
        self.trait_set(trait_change_notify=False,
                 ARcenterC = self.HRcenterC - self.normVectHR * self.thickness)
        self.trait_set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)

    def _wedgeAngle_changed(self, old, new):
        self.trait_set(trait_change_notify=False,
                 normAngleAR = np.mod(self.normAngleHR + pi + self.wedgeAngle, 2*pi))
        self.trait_set(trait_change_notify=False,
                 normVectAR = optics.geometric.vector_rotation_2D(self.normVectHR, pi+self.wedgeAngle))
        self.trait_set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR * self.sagAR)

    def _inv_ROC_HR_changed(self, old, new):
        #First update the sag
        if np.abs(self.inv_ROC_HR) > 1./(10*km):
            R = 1./self.inv_ROC_HR
            r = self.diameter/2
            self.sagHR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagHR = 0.0;

        #Then move whichever end of the sagitta anchor_point does not pin.
        if self.anchor_point == 'HRcenter':
            #The arc stays under the beam and the substrate moves back
            #behind it. The notification is what carries it: suppressing
            #it left ARcenterC, ARcenter and center where the old
            #sagitta had put them. _HRcenterC_changed cannot come back
            #round - every assignment it makes is silent - and the
            #HRcenter it recomputes is the inverse of the line below, so
            #the fixed point survives the trip.
            self.HRcenterC = self.HRcenter - self.sagHR*self.normVectHR
        else:
            #The substrate stays where it was put and the arc moves on
            #it. Nothing else needs saying: the chord planes, and so the
            #centre, are untouched.
            self.trait_set(trait_change_notify=False,
                     HRcenter = self.HRcenterC + self.sagHR*self.normVectHR)

    def _inv_ROC_AR_changed(self, old, new):
        #First update the sag
        if np.abs(self.inv_ROC_AR) > 1./(10*km):
            R = 1./self.inv_ROC_AR
            r = self.diameter/2
            self.sagAR =  - np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))
        else:
            self.sagAR = 0.0;
        #The AR arc stands on ARcenterC, which the thickness fixes, so
        #nothing moves but the apex. There is no position written here
        #for a notification to ride on, unlike the HR above, so this one
        #has to be explicit.
        self.trait_set(trait_change_notify=False,
                 ARcenter = self.ARcenterC + self.normVectAR*self.sagAR)

    def _diameter_changed(self, old, new):
        #The sag of a curved surface depends on the aperture as well as
        #on the ROC, so a new diameter changes it. Recompute both sags
        #the way a new curvature would, leaving a diameter change and a
        #curvature change with the same consequences.
        self._inv_ROC_HR_changed(self.inv_ROC_HR, self.inv_ROC_HR)
        self._inv_ROC_AR_changed(self.inv_ROC_AR, self.inv_ROC_AR)

#}}}

#}}}

#{{{ Cylindrical Mirror Class

class CyMirror(Mirror):
    '''
    Representing a partial reflective cylindrical mirror. Note that both HR and AR surfaces are treated as cylindrical if you specify non-zero ROC for them. The curve  directions of the two surfaces must be the same.

    A cylinder focuses in one plane and does nothing in the other, so a
    beam leaving one of these is astigmatic even at normal incidence.
    Away from normal incidence the two directions are not a relabelling
    of each other: a surface of radius R at incidence theta presents an
    effective radius R*cos(theta) in the plane of incidence and
    R/cos(theta) perpendicular to it, so 'h' gives a focal length of
    R*cos(theta)/2 and 'v', of the same radius, gives R/(2*cos(theta)).

    Only 'h' is visible in the drawing. What the plane of the trace cuts
    out of a 'v' cylinder is a straight line, so the faces are drawn
    straight and the focusing happens entirely out of the page.

    The ray matrices are Siegman, Lasers, Table 15.1, with the curvature
    given to one plane and zero to the other; see cyl_refl_defl_angle.
    Note that only in reflection does the uncurved plane come out as the
    identity. In transmission it keeps the tilt scaling and the index
    change, and loses the power alone.

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
        The position of the apex of the arc of the AR surface. shape(2,).
        Note that this is the counterpart of HRcenter, not of HRcenterC:
        it lies one sagitta out of the substrate. Anything that asks for
        the centre of a chord - line_arc_intersection(), or an arc drawn
        from its own chord - wants ARcenterC instead.
    ARcenterC : array
        The position of the center of the chord of the AR surface.
        shape(2,).
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
    HRreflective : boolean
        A boolean value defaults to True. If False, the HR surface of
        this optics is not supposed to reflect beams. Therefore, every
        reflection at the HR surface, from outside or from inside the
        substrate, increases the stray_order: such a reflection is a
        ghost. The mirror image of HRtransmissive. This flag should be
        set to False for lenses.
    term_on_HR : boolean
        If this is True, a beam with stray_order <= self.term_on_HR_order will be terminated when
        it hits on HR. This is to avoid the inifinite loop of non-sequencial
        trace by forming a cavity.
    term_on_HR_order : int
        Integer to specify the upper limit of the stray order used to judge
        whether to terminate the non sequential trace or not on HR reflection.
    term_on_HR_transmits : boolean
        What term_on_HR stops. False, the default, stops the beam at the
        surface: nothing is computed, which is what term_on_HR has always
        done. True stops only the reflection that would form the cavity,
        and lets the element be hit as usual otherwise, so the beam
        transmitted through the substrate carries on and the ghosts
        inside it are unfolded, counted and capped by order and
        max_stray_order like any others. Only the external reflection
        off the HR is dropped; a ghost leaving through the HR from
        inside the substrate is not. Has no effect unless term_on_HR is
        True, as term_on_HR_order has none.
    '''

#{{{ __init__

    def __init__(self, HRcenter=[0.0,0.0], normAngleHR=0.0,
                 normVectHR=None, diameter=25.0*cm, thickness=15.0*cm,
                 wedgeAngle=0.25*pi/180., inv_ROC_HR=1.0/7000.0, inv_ROC_AR=0.0,
                 Refl_HR=0.99, Trans_HR=0.01, Refl_AR=0.01, Trans_AR=0.99, n=1.45,
                 name="Mirror", HRtransmissive=False, HRreflective=True,
                 term_on_HR=False, term_on_HR_order=0,
                 term_on_HR_transmits=False,
                 max_stray_order=None, curve_direction='h'):
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
        HRreflective : boolean, optional
            If False, every reflection at the HR surface, from either
            side, increases the stray_order: this optics is not
            supposed to reflect there, so such a reflection is a ghost.
            Set to False for lenses.
            Defaults True
        term_on_HR : boolean, optional
            If this is True, a beam with stray_order <= self.term_on_HR_order
            will be terminated when
            it hits on HR. This is to avoid the inifinite loop of
            non-sequencial
            trace by forming a cavity.
            Defaults False.
        term_on_HR_order : int, optional
            Upper limit of the stray order at which term_on_HR still
            terminates a beam. A beam of a higher stray order is left
            alone.
            Defaults 0.
        term_on_HR_transmits : boolean, optional
            What term_on_HR stops. False stops the beam at the
            surface and computes nothing. True stops only the
            external reflection off the HR, so the beam transmitted
            through the substrate carries on. See the class
            docstring.
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
        self.HRreflective = HRreflective
        self.term_on_HR = term_on_HR
        self.term_on_HR_order = term_on_HR_order
        self.term_on_HR_transmits = term_on_HR_transmits
        self.max_stray_order = max_stray_order
        self.curve_direction = curve_direction

#}}}

#{{{ copy

    def copy(self):
        m = CyMirror(HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                      diameter=self.diameter, thickness=self.thickness,
                      wedgeAngle=self.wedgeAngle, inv_ROC_HR=self.inv_ROC_HR,
                      inv_ROC_AR=self.inv_ROC_AR, Refl_HR=self.Refl_HR,
                      Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR, Trans_AR=self.Trans_AR,
                      n=self.n, name=self.name, HRtransmissive=self.HRtransmissive,
                      HRreflective=self.HRreflective,
                      term_on_HR=self.term_on_HR,
                      term_on_HR_order=self.term_on_HR_order,
                      term_on_HR_transmits=self.term_on_HR_transmits,
                      max_stray_order=self.max_stray_order,
                      curve_direction=self.curve_direction)
        m.anchor_point = self.anchor_point
        return m

#}}}

#{{{ get_side_info

    def _geometry_key(self):
        '''
        Mirror's key, and which way the cylinder is turned: that is what
        decides whether the faces meet the sides at their chords or at
        their apexes. See Mirror._geometry_key.
        '''
        return Mirror._geometry_key(self) + (self.curve_direction,)

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

#{{{ get_corners

    def get_corners(self):
        '''
        The four corners of the substrate. See Mirror.get_corners.

        A cylinder curved across the plane of the drawing has no
        curvature in it, so its faces meet the sides at their apexes and
        the substrate is that much thicker in section. That is the same
        distinction get_side_info() makes, and for the same reason.
        '''
        if self.curve_direction == 'v':
            center_of_HR = self.HRcenter
            thickness = self.thickness + self.sagHR + self.sagAR
        else:
            center_of_HR = self.HRcenterC
            thickness = self.thickness

        plVect = optics.geometric.vector_rotation_2D(self.normVectHR, pi/2)
        p1 = center_of_HR + plVect * self.diameter/2
        p2 = p1 - plVect * self.diameter
        p3 = p2 - self.normVectHR * (thickness
                                     - np.tan(self.wedgeAngle)*self.diameter/2)
        p4 = p1 - self.normVectHR * (thickness
                                     + np.tan(self.wedgeAngle)*self.diameter/2)
        return [p1, p2, p3, p4]



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

        #See Mirror.isHit.
        if self._misses_bounding_circle(beam):
            return {'isHit': False, 'position': np.array((0., 0.)),
                    'distance': 0.0, 'face': ''}

        if self.curve_direction == 'h':
            HRsurface = {'center': self.HRcenterC, 'normal_vector': self.normVectHR,
                         'size': self.diameter, 'inv_ROC': self.inv_ROC_HR, 'name': 'HR'}
            ARsurface = {'center': self.ARcenterC, 'normal_vector': self.normVectAR,
                             'size': self.ARdiameter, 'inv_ROC': self.inv_ROC_AR, 'name': 'AR'}
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

        if self.draw_HR_marker:
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
            #On the chord centre, not the apex. See Mirror.draw().
            v = v.T + self.ARcenterC
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

        A beam is produced as long as its stray_order is not above
        ``order`` and its power is over the threshold. The stray_order
        of the incident beam is carried over, so a beam that is already
        stray leaves less of the allowance for the ghosts made here.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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

        #The shape of the surface in the plane of the trace and the
        #power of the surface are two different things here, and they
        #agree only when the curvature lies in that plane. Curved out of
        #the plane, the cross-section the trace sees is a straight line,
        #so the geometry is flat - but the surface still focuses, out of
        #the plane. The *_geom values answer the geometric question,
        #which is where the beam lands and how its envelope is drawn;
        #self.inv_ROC_* carries the power, and cyl_refl_defl_angle
        #decides which plane receives it.
        if self.curve_direction == 'h':
            chord_center_HR = self.HRcenterC
            chord_center_AR = self.ARcenterC
            inv_ROC_HR_geom = self.inv_ROC_HR
            inv_ROC_AR_geom = self.inv_ROC_AR
        else:
            chord_center_HR = self.HRcenter
            chord_center_AR = self.ARcenter
            inv_ROC_HR_geom = 0.0
            inv_ROC_AR_geom = 0.0

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=chord_center_HR,
                                                     chordNormVect=self.normVectHR,
                                                     invROC=inv_ROC_HR_geom,
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
        beam_in.incSurfInvROC = inv_ROC_HR_geom
        beams['input']= beam_in


        #Propagate the input beam to the intersection point
        beam_on_HR = beam_in.copy()
        beam_on_HR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.cyl_refl_defl_angle(
                        beam_on_HR.dirAngle, localNormAngle, 1.0, self.n,
                        invROC=self.inv_ROC_HR,
                        curve_direction=self.curve_direction)
        #Reflected beam
        beam_r1 = beam_on_HR.copy()
        beam_r1.P = beam_r1.P * self.Refl_HR
        #A face meant to reflect makes no ghost here: the reflection is
        #what the face is for, so the beam leaves at the stray order it
        #arrived at and `order` - which counts ghost generations - has
        #nothing to say about it. An already-stray beam reflects off a
        #mirror like any other. A face not meant to reflect does make a
        #ghost, and that one is counted and capped.
        ghost = not self.HRreflective
        if ghost:
            beam_r1.stray_order = beam_r1.stray_order+1
        if beam_r1.P > threshold and not (ghost
                                          and beam_r1.stray_order > order):
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = inv_ROC_HR_geom
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
        beam_s1.departSurfInvROC = -inv_ROC_HR_geom

        #Hit AR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=chord_center_AR,
                                                     chordNormVect=-self.normVectAR,
                                                     invROC=-inv_ROC_AR_geom,
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
        beam_s1.incSurfInvROC = -inv_ROC_AR_geom
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the AR surface
        beam_on_AR = beam_s1.copy()
        beam_on_AR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.cyl_refl_defl_angle(
                        beam_on_AR.dirAngle, localNormAngle, self.n, 1.0,
                        invROC=-self.inv_ROC_AR,
                        curve_direction=self.curve_direction)

        #Transmitted beam
        beam_t1 = beam_on_AR.copy()
        beam_t1.P = beam_on_AR.P * self.Trans_AR
        if beam_t1.P > threshold:
            beam_t1.dirAngle = deflAngle
            beam_t1.n = 1.0
            beam_t1.ABCDTrans(Mtx, Mty)
            beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
            beam_t1.departSurfInvROC = inv_ROC_AR_geom
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
        beam_sr.departSurfInvROC = -inv_ROC_AR_geom


        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit the HR from the back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=chord_center_HR,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-inv_ROC_HR_geom,
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
            beam_sr.incSurfInvROC = -inv_ROC_HR_geom
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr

            #Propagate the input beam to the intersection point
            beam_on_HR = beam_sr.copy()
            beam_on_HR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.cyl_refl_defl_angle(
                            beam_on_HR.dirAngle, localNormAngle, self.n, 1.0,
                            invROC=-self.inv_ROC_HR,
                            curve_direction=self.curve_direction)

            #Transmitted through HR
            beam_r1 = beam_on_HR.copy()
            beam_r1.P = beam_r1.P * self.Trans_HR
            beam_r1.stray_order = beam_r1.stray_order+1
            if beam_r1.P > threshold and beam_r1.stray_order <= order:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = inv_ROC_HR_geom
                beam_r1.incSurfAngle = None
                beam_r1.incSurfInvROC = None
                beam_r1.name = self.name+':r%d'%(ii+1)
                beams['r'+str(ii+1)] = beam_r1

            #Reflected by HR
            beam_s1 = beam_on_HR.copy()
            beam_s1.P = beam_s1.P * self.Refl_HR
            if not self.HRreflective:
                beam_s1.stray_order = beam_s1.stray_order+1
            if beam_s1.P < threshold or beam_s1.stray_order > order:
                break
            beam_s1.dirAngle = reflAngle
            beam_s1.ABCDTrans(Mrx, Mry)
            beam_s1.departSurfAngle = localNormAngle
            beam_s1.departSurfInvROC = -inv_ROC_HR_geom

            #Hit AR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=chord_center_AR,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-inv_ROC_AR_geom,
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
            beam_s1.incSurfInvROC = -inv_ROC_AR_geom
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1

            #Propagate the beam to the AR surface
            beam_on_AR = beam_s1.copy()
            beam_on_AR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.cyl_refl_defl_angle(
                            beam_on_AR.dirAngle, localNormAngle, self.n, 1.0,
                            invROC=-self.inv_ROC_AR,
                            curve_direction=self.curve_direction)
            #Transmitted beam
            beam_t1 = beam_on_AR.copy()
            beam_t1.P = beam_on_AR.P * self.Trans_AR
            if beam_t1.P > threshold:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = inv_ROC_AR_geom
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
            beam_sr.departSurfInvROC = -inv_ROC_AR_geom

            ii=ii+1

        return beams
#}}}

#{{{ hitFromAR

    def hitFromAR(self, beam, order=0, threshold=0.0, verbose=False):
        '''
        Compute the reflected and deflected beams when
        an input beam hit the AR surface.

        A beam is produced as long as its stray_order is not above
        ``order`` and its power is over the threshold. The stray_order
        of the incident beam is carried over, so a beam that is already
        stray leaves less of the allowance for the ghosts made here.

        Parameters
        ----------
        beam : gtrace.beam.GaussianBeam
            A GaussianBeam object to be interacted by the optics.
        order : int, optional
            The largest stray_order a beam produced here may have.
            The count is carried over from the incident beam, not
            started afresh.
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

        #The shape of the surface in the plane of the trace and the
        #power of the surface are two different things here, and they
        #agree only when the curvature lies in that plane. Curved out of
        #the plane, the cross-section the trace sees is a straight line,
        #so the geometry is flat - but the surface still focuses, out of
        #the plane. The *_geom values answer the geometric question,
        #which is where the beam lands and how its envelope is drawn;
        #self.inv_ROC_* carries the power, and cyl_refl_defl_angle
        #decides which plane receives it.
        if self.curve_direction == 'h':
            chord_center_HR = self.HRcenterC
            chord_center_AR = self.ARcenterC
            inv_ROC_HR_geom = self.inv_ROC_HR
            inv_ROC_AR_geom = self.inv_ROC_AR
        else:
            chord_center_HR = self.HRcenter
            chord_center_AR = self.ARcenter
            inv_ROC_HR_geom = 0.0
            inv_ROC_AR_geom = 0.0

        #Get the intersection point
        ans = optics.geometric.line_arc_intersection(pos=beam.pos, dirVect=beam.dirVect,
                                                     chord_center=chord_center_AR,
                                                     chordNormVect=self.normVectAR,
                                                     invROC=inv_ROC_AR_geom,
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
        beam_in.incSurfInvROC = inv_ROC_AR_geom
        beam_in.length = ans['distance']
        beams['input']= beam_in

        #Propagate the input beam to the intersection point
        beam_on_AR = beam_in.copy()
        beam_on_AR.propagate(ans['distance'])

        #Calculate reflection and deflection angles along with the ABCD matrices
        #for reflection and deflection.
        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.cyl_refl_defl_angle(
                        beam_on_AR.dirAngle, localNormAngle, 1.0, self.n,
                        invROC=self.inv_ROC_AR,
                        curve_direction=self.curve_direction)
        #Reflected beam
        beam_r1 = beam_on_AR.copy()
        beam_r1.P = beam_r1.P * self.Refl_AR
        beam_r1.stray_order = beam_r1.stray_order+1
        if beam_r1.P > threshold and beam_r1.stray_order <= order:
            beam_r1.dirAngle = reflAngle
            beam_r1.ABCDTrans(Mrx, Mry)
            beam_r1.departSurfAngle = localNormAngle
            beam_r1.departSurfInvROC = inv_ROC_AR_geom
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
        beam_s1.departSurfInvROC = -inv_ROC_AR_geom

        #Hit HR from back
        ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                     chord_center=chord_center_HR,
                                                     chordNormVect=-self.normVectHR,
                                                     invROC=-inv_ROC_HR_geom,
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
        beam_s1.incSurfInvROC = -inv_ROC_HR_geom
        beam_s1.name = self.name+':s1'
        beams['s1'] = beam_s1


        #Propagate the beam to the HR surface
        beam_on_HR = beam_s1.copy()
        beam_on_HR.propagate(ans['distance'])

        (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                    optics.geometric.cyl_refl_defl_angle(
                        beam_on_HR.dirAngle, localNormAngle, self.n, 1.0,
                        invROC=-self.inv_ROC_HR,
                        curve_direction=self.curve_direction)

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
            beam_t1.departSurfInvROC = inv_ROC_HR_geom
            beam_t1.incSurfAngle = None
            beam_t1.incSurfInvROC = None
            beam_t1.name = self.name+':t1'
            beams['t1'] = beam_t1

        #Reflected beam
        beam_sr = beam_on_HR.copy()
        beam_sr.P = beam_sr.P * self.Refl_HR
        if not self.HRreflective:
            beam_sr.stray_order = beam_sr.stray_order+1
        if beam_sr.P < threshold or beam_sr.stray_order > order:
            return beams
        beam_sr.dirAngle = reflAngle
        beam_sr.ABCDTrans(Mrx, Mry)
        beam_sr.departSurfAngle = localNormAngle
        beam_sr.departSurfInvROC = -inv_ROC_HR_geom

        #Calculate higher order reflections

        ii = 1
        while ii <= 10*order:

            #Hit AR from back

            #Get the intersection point
            ans = optics.geometric.line_arc_intersection(pos=beam_sr.pos, dirVect=beam_sr.dirVect,
                                                         chord_center=chord_center_AR,
                                                         chordNormVect=-self.normVectAR,
                                                         invROC=-inv_ROC_AR_geom,
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
            beam_sr.incSurfInvROC = -inv_ROC_AR_geom
            beam_sr.name = self.name+':s%d'%(2*ii)
            beams['s'+str(2*ii)]= beam_sr


            #Propagate the input beam to the intersection point
            beam_on_AR = beam_sr.copy()
            beam_on_AR.propagate(ans['distance'])

            #Calculate reflection and deflection angles along with the ABCD matrices
            #for reflection and deflection.
            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.cyl_refl_defl_angle(
                            beam_on_AR.dirAngle, localNormAngle, self.n, 1.0,
                            invROC=-self.inv_ROC_AR,
                            curve_direction=self.curve_direction)

            #Transmitted through AR
            beam_r1 = beam_on_AR.copy()
            beam_r1.P = beam_r1.P * self.Trans_AR
            if beam_r1.P > threshold:
                beam_r1.dirAngle = deflAngle
                beam_r1.n = 1.0
                beam_r1.ABCDTrans(Mtx, Mty)
                beam_r1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_r1.departSurfInvROC = inv_ROC_AR_geom
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
            beam_s1.departSurfInvROC = -inv_ROC_AR_geom

            #Hit HR from back
            ans = optics.geometric.line_arc_intersection(pos=beam_s1.pos, dirVect=beam_s1.dirVect,
                                                         chord_center=chord_center_HR,
                                                         chordNormVect=-self.normVectHR,
                                                         invROC=-inv_ROC_HR_geom,
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
            beam_s1.incSurfInvROC = -inv_ROC_HR_geom
            beam_s1.length = ans['distance']
            beam_s1.name = self.name+':s%d'%(2*ii+1)
            beams['s'+str(2*ii+1)] = beam_s1


            #Propagate the beam to the HR surface
            beam_on_HR = beam_s1.copy()
            beam_on_HR.propagate(ans['distance'])

            (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) = \
                        optics.geometric.cyl_refl_defl_angle(
                            beam_on_HR.dirAngle, localNormAngle, self.n, 1.0,
                            invROC=-self.inv_ROC_HR,
                            curve_direction=self.curve_direction)

            #Transmitted beam
            beam_t1 = beam_on_HR.copy()
            beam_t1.P = beam_t1.P * self.Trans_HR
            beam_t1.stray_order = beam_t1.stray_order+1
            if beam_t1.P > threshold and beam_t1.stray_order <= order:
                beam_t1.dirAngle = deflAngle
                beam_t1.n = 1.0
                beam_t1.ABCDTrans(Mtx, Mty)
                beam_t1.departSurfAngle = np.mod(localNormAngle+pi, 2*pi)
                beam_t1.departSurfInvROC = inv_ROC_HR_geom
                beam_t1.incSurfAngle = None
                beam_t1.incSurfInvROC = None
                beam_t1.name = self.name+':t%d'%(ii+1)
                beams['t'+str(ii+1)] = beam_t1

            #Reflected beam
            beam_sr = beam_on_HR.copy()
            beam_sr.P = beam_sr.P * self.Refl_HR
            if not self.HRreflective:
                beam_sr.stray_order = beam_sr.stray_order+1
            if beam_sr.P < threshold or beam_sr.stray_order > order:
                break
            beam_sr.dirAngle = reflAngle
            beam_sr.ABCDTrans(Mrx, Mry)
            beam_sr.departSurfAngle = localNormAngle
            beam_sr.departSurfInvROC = -inv_ROC_HR_geom

            ii=ii+1

        return beams

#}}}

#}}}

#{{{ Lens Class

class LensGeometryError(ValueError):
    '''
    Raised when the lens asked for cannot be made out of the blank it
    was given: a face steeper than its own aperture, two concave faces
    that would meet in the middle, or a focal length no substrate of
    that thickness can reach.

    A subclass of ValueError, so that code catching the ordinary bad
    argument still catches these.
    '''
    pass

#: Curvature below which Mirror's sagitta bookkeeping calls a face flat
#: (a radius over 10 km). Kept in step with _inv_ROC_HR_changed so that
#: the sagitta computed here is the one the traits carry.
_FLAT_CURVATURE = 1./(10*km)

#: Lens shapes, spelled the way a catalogue spells them: the HR face
#: first, the AR face second, so 'plano-convex' is flat towards the
#: front and convex towards the back, and 'convex-plano' is the same
#: lens turned round.
#:
#: 'symmetric' and 'meniscus' name a family without saying which way it
#: curves, and let the sign of f decide. The spelt-out names are the
#: same families with each face pinned, so that asking for a biconvex
#: lens of negative focal length is an error instead of a surprise.
#:
#: Each entry is (family, HR face, AR face).
_LENS_SHAPES = {
    'symmetric':      ('symmetric', None, None),
    'biconvex':       ('symmetric', 'convex', 'convex'),
    'biconcave':      ('symmetric', 'concave', 'concave'),
    'plano-convex':   ('plano', 'plano', 'convex'),
    'convex-plano':   ('plano', 'convex', 'plano'),
    'plano-concave':  ('plano', 'plano', 'concave'),
    'concave-plano':  ('plano', 'concave', 'plano'),
    'meniscus':       ('meniscus', None, None),
    'convex-concave': ('meniscus', 'convex', 'concave'),
    'concave-convex': ('meniscus', 'concave', 'convex'),
}

#: Iterations allowed while the centre thickness and the curvatures
#: settle on each other. The dependence is weak - the thickness moves by
#: a sagitta - so this converges in a handful; the count is a backstop.
_SOLVE_ITERATIONS = 100

#{{{ Lens geometry, as free functions

def sagitta(inv_ROC, diameter):
    '''
    Sagitta of a spherical face spanning an aperture.

    Parameters
    ----------
    inv_ROC : float
        1/ROC of the face. Positive for a concave face, as everywhere
        else in gtrace.
    diameter : float
        Aperture the face has to span.

    Returns
    -------
    float
        The sagitta, positive when the face bulges out of the substrate
        and negative when it is hollowed into it. Zero for a face
        gtrace treats as flat.

    Raises
    ------
    LensGeometryError
        If the radius is smaller than the semi-aperture, in which case
        no arc of that radius reaches across the face. Left to run,
        this is where the square root in Mirror's sagitta goes complex
        and NaN starts spreading through the geometry.
    '''
    inv_ROC = float(inv_ROC)
    if np.abs(inv_ROC) <= _FLAT_CURVATURE:
        return 0.0
    R = 1./inv_ROC
    r = diameter/2.
    if np.abs(R) < r:
        raise LensGeometryError(
            'A face of ROC %.6g m cannot span an aperture of %.6g m: the '
            'radius has to be at least the semi-aperture, %.6g m.'
            % (R, diameter, r))
    return -np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))

def lens_power(inv_ROC_HR, inv_ROC_AR, center_thickness, n):
    '''
    Optical power 1/f of a thick lens, in gtrace's sign convention.

    The lensmaker's equation with the thickness term, rewritten in terms
    of the inverse radii gtrace stores. Those are positive for a concave
    face on both sides of the substrate, where the usual convention
    measures both radii along the direction of propagation, so
    1/R1 = -inv_ROC_HR and 1/R2 = +inv_ROC_AR.

    Parameters
    ----------
    inv_ROC_HR, inv_ROC_AR : float
        1/ROC of the two faces.
    center_thickness : float
        Distance between the two apexes, which is what the thickness
        term means. For a Mirror-like substrate this is
        ``thickness + sagHR + sagAR``, not ``thickness``.
    n : float
        Index of refraction.

    Returns
    -------
    float
        1/f. Positive for a converging lens.
    '''
    c1 = float(inv_ROC_HR)
    c2 = float(inv_ROC_AR)
    return (n - 1.)*(-(c1 + c2) - (n - 1.)*center_thickness*c1*c2/n)

def _lens_shape(shape):
    '''
    Look a shape name up, defaulting to the symmetric family.
    '''
    if shape is None:
        shape = 'symmetric'
    try:
        return _LENS_SHAPES[shape]
    except (KeyError, TypeError):
        raise LensGeometryError(
            'Unknown lens shape %r. Choose one of: %s. The two-part names '
            'are spelt HR face first, AR face second.'
            % (shape, ', '.join(sorted(_LENS_SHAPES))))

def _check_center_thickness(thickness, sag_HR, sag_AR):
    '''
    Refuse a blank the two faces would eat through.

    Raises
    ------
    LensGeometryError
        If nothing is left in the middle.
    '''
    center_thickness = thickness + sag_HR + sag_AR
    if center_thickness > 0.:
        return center_thickness
    raise LensGeometryError(
        'The two faces would meet inside the substrate: a rim thickness '
        'of %.6g mm with sagittae of %.6g and %.6g mm leaves a centre '
        'thickness of %.6g mm. This lens needs a rim thickness over '
        '%.6g mm.'
        % (thickness/mm, sag_HR/mm, sag_AR/mm, center_thickness/mm,
           -(sag_HR + sag_AR)/mm))

def _check_lens_inputs(f, n, diameter, thickness):
    '''
    Reject the arguments no lens can be made from, whatever its shape.

    Returns
    -------
    float
        The focal length, as a float.
    '''
    f = float(f)
    if f == 0. or not np.isfinite(f):
        raise LensGeometryError(
            'A lens needs a finite, non-zero focal length; got %r. A flat '
            'substrate is a Mirror with both curvatures zero.' % (f,))
    if n <= 1.:
        raise LensGeometryError(
            'A lens needs a substrate denser than its surroundings; got '
            'n = %r.' % (n,))
    if diameter <= 0. or thickness <= 0.:
        raise LensGeometryError(
            'diameter and thickness must be positive; got %r and %r.'
            % (diameter, thickness))
    return f

def _face_kind(inv_ROC):
    '''
    Whether a face is convex, concave or flat, in the vocabulary the
    shape names use.
    '''
    if np.abs(inv_ROC) <= _FLAT_CURVATURE:
        return 'plano'
    return 'concave' if inv_ROC > 0 else 'convex'

def _curvatures_at(f, family, flat_face, n, center_thickness, fixed):
    '''
    The two curvatures giving focal length f, for a centre thickness
    held fixed. solve_lens_curvatures() iterates on the thickness
    around this.
    '''
    if family == 'plano':
        #One face flat kills the thickness term outright, so this is
        #exact rather than a starting point.
        c = -1./((n - 1.)*f)
        return (0.0, c) if flat_face == 'HR' else (c, 0.0)

    if family == 'symmetric':
        #Both faces the same: a quadratic in the shared curvature.
        A = (n - 1.)**2*center_thickness/n
        B = 2.*(n - 1.)
        C = 1./f
        if A <= 0.:
            c = -C/B
            return c, c
        disc = B*B - 4.*A*C
        if disc < 0.:
            raise LensGeometryError(
                'No symmetric lens of f = %.6g m can be made from a blank '
                '%.6g m thick with n = %.6g. Once the substrate is that '
                'thick relative to the focal length the two faces stop '
                'being able to reach it; f would have to exceed d/n = '
                '%.6g m.' % (f, center_thickness, n, center_thickness/n))
        #Of the two roots this is the weakly curved one; the other bends
        #the faces back on themselves and would fail the aperture check.
        c = (-B + np.sqrt(disc))/(2.*A)
        return c, c

    #Meniscus: one face is given, the other follows linearly.
    k = (n - 1.)*center_thickness/n
    denom = 1. + k*fixed
    if np.abs(denom) < 1e-12:
        raise LensGeometryError(
            'A meniscus with an HR face of ROC %.6g m has no solution for '
            'the AR face: at this curvature the substrate cancels the '
            'face it stands on.' % (1./fixed,))
    c_AR = -(1./((n - 1.)*f) + fixed)/denom
    return fixed, c_AR

def solve_lens_curvatures(f, shape=None, diameter=1*inch, thickness=6*mm,
                          n=1.45, inv_ROC_HR=None):
    '''
    Curvatures of the two faces of a lens of focal length f.

    Solved as a thick lens, which is what gtrace then traces: the beam
    is refracted at both faces with the substrate in between, so a
    curvature taken from the thin lens formula would come out a few
    parts in a thousand away from the focal length asked for. The
    centre thickness the thickness term needs itself depends on the
    curvatures through the sagittae, so the two are iterated to
    convergence. For a plano lens the thickness term vanishes and the
    answer is exact in one step.

    Parameters
    ----------
    f : float
        Focal length. Positive converges.
    shape : str or None, optional
        One of the names in _LENS_SHAPES, spelt HR face first. None
        means the symmetric family: biconvex for a positive f,
        biconcave for a negative one.
    diameter : float, optional
        Aperture the faces have to span. Defaults 1 inch.
    thickness : float, optional
        Distance between the two chord planes, which is Mirror's
        thickness and the thickness at the rim. Defaults 6 mm.
    n : float, optional
        Index of refraction. Defaults 1.45.
    inv_ROC_HR : float or None, optional
        1/ROC of the HR face, for a meniscus, which f alone does not
        determine. Not accepted for the other families, where it is
        solved for.

    Returns
    -------
    (float, float)
        ``(inv_ROC_HR, inv_ROC_AR)``.

    Raises
    ------
    LensGeometryError
        If the lens cannot be made: a face steeper than its aperture,
        a centre eaten through by two concave faces, a focal length out
        of reach of the substrate, or a shape contradicting the sign of
        f.
    '''
    family, want_HR, want_AR = _lens_shape(shape)
    f = _check_lens_inputs(f, n, diameter, thickness)

    if family == 'meniscus' and inv_ROC_HR is None:
        raise LensGeometryError(
            'A meniscus is not determined by f alone: a whole family of '
            'them has the same focal length. Give ROC_HR (or inv_ROC_HR) '
            'to pin the front face, and the back face is solved for.')
    if family != 'meniscus' and inv_ROC_HR is not None:
        raise LensGeometryError(
            'Both faces of a %s lens are solved for from f. Only a '
            'meniscus takes one of its radii as an input.'
            % (shape or 'symmetric',))

    flat_face = 'HR' if want_HR == 'plano' else 'AR'
    fixed = None if inv_ROC_HR is None else float(inv_ROC_HR)

    #Iterate the centre thickness and the curvatures onto each other.
    #Starting from the rim thickness, which is off by the sagittae.
    d = thickness
    c_HR, c_AR = _curvatures_at(f, family, flat_face, n, d, fixed)
    for _ in range(_SOLVE_ITERATIONS):
        c_HR, c_AR = _curvatures_at(f, family, flat_face, n, d, fixed)
        d_next = (thickness + sagitta(c_HR, diameter)
                  + sagitta(c_AR, diameter))
        if np.abs(d_next - d) <= 1e-15*max(1., np.abs(d_next)):
            d = d_next
            break
        d = d_next
    c_HR, c_AR = _curvatures_at(f, family, flat_face, n, d, fixed)

    #Whether the blank survives the faces comes first: with a negative
    #centre thickness the iteration above is solving for a substrate
    #that does not exist, and the residual check below would report
    #that as a solver failure rather than as the impossible lens it is.
    d = _check_center_thickness(thickness, sagitta(c_HR, diameter),
                                sagitta(c_AR, diameter))

    #What came out has to actually have the focal length asked for.
    #This catches a solve that wandered as well as one that never
    #converged, and it costs nothing.
    got = lens_power(c_HR, c_AR, d, n)
    if np.abs(got - 1./f) > 1e-9*np.abs(1./f):
        raise LensGeometryError(
            'The curvatures did not settle on f = %.6g m (they give '
            '%.6g m). This is a bug in the solver, not in the lens asked '
            'for.' % (f, 1./got if got else np.inf))

    for face, want, c in [('HR', want_HR, c_HR), ('AR', want_AR, c_AR)]:
        if want is not None and _face_kind(c) != want:
            raise LensGeometryError(
                'A %r lens wants a %s %s face, but f = %.6g m makes it %s. '
                'A converging lens is convex where a diverging one is '
                'concave; either change the shape or the sign of f.'
                % (shape, want, face, f, _face_kind(c)))

    #A meniscus curves the same way throughout, which in this sign
    #convention - each face measured from its own side - means the two
    #curvatures have opposite signs. Pinning the front face too gently
    #puts the solution outside that family, and asking for a meniscus
    #and quietly getting a biconvex lens is exactly what naming the
    #shape is supposed to prevent.
    if family == 'meniscus' and c_HR*c_AR >= 0:
        #c_AR passes through zero when the front face alone carries the
        #whole power, so the boundary is exact rather than thin-lens.
        boundary = (n - 1.)*np.abs(f)
        raise LensGeometryError(
            'ROC_HR = %.6g m at f = %.6g m comes out %s, not a meniscus. '
            'A front face curving the same way as the lens as a whole '
            'makes a meniscus only while |ROC_HR| < (n-1)|f| = %.6g m; '
            'curving it the other way always does.'
            % (1./c_HR if c_HR else np.inf, f,
               '%s-%s' % (_face_kind(c_HR), _face_kind(c_AR)), boundary))

    return c_HR, c_AR

def rescale_lens_curvatures(inv_ROC_HR, inv_ROC_AR, f, diameter=1*inch,
                            thickness=6*mm, n=1.45):
    '''
    Scale both curvatures by one common factor until the lens has focal
    length f.

    This is how a lens changes focal length without changing shape. The
    ratio between the two faces is what makes a lens biconvex or
    plano-convex or a meniscus, and multiplying both by the same number
    leaves that ratio alone: a flat face stays flat, an equiconvex lens
    stays equiconvex, a meniscus keeps its bend. A negative factor
    turns the whole lens inside out, which is what asking a converging
    lens for a negative focal length means.

    Parameters
    ----------
    inv_ROC_HR, inv_ROC_AR : float
        The curvatures to scale. At least one must be non-zero: a flat
        substrate has no shape to keep.
    f : float
        The focal length wanted.
    diameter, thickness, n : float, optional
        The substrate, as for solve_lens_curvatures().

    Returns
    -------
    (float, float)
        ``(inv_ROC_HR, inv_ROC_AR)``, scaled.

    Raises
    ------
    LensGeometryError
        If no scaling of this shape reaches that focal length, or if
        the one that does cannot be cut from the substrate.
    '''
    c1 = float(inv_ROC_HR)
    c2 = float(inv_ROC_AR)
    f = _check_lens_inputs(f, n, diameter, thickness)

    if c1 == 0. and c2 == 0.:
        raise LensGeometryError(
            'A substrate with two flat faces has no shape to scale. Say '
            'which shape the lens should take as well as its focal '
            'length.')

    #Power as a function of the scale s is a s^2 + b s, and at least one
    #of a and b is non-zero once the two faces are not both flat.
    C = 1./f
    b = -(n - 1.)*(c1 + c2)
    #Thin lens estimate, and the branch to stay on: as the substrate
    #thins the physical root goes to this one.
    s0 = C/b if b else 0.0

    d = thickness
    s = s0
    for _ in range(_SOLVE_ITERATIONS):
        a = -(n - 1.)**2*d*c1*c2/n
        s = _scale_root(a, b, C, s0, f)
        d_next = (thickness + sagitta(s*c1, diameter)
                  + sagitta(s*c2, diameter))
        if np.abs(d_next - d) <= 1e-15*max(1., np.abs(d_next)):
            d = d_next
            break
        d = d_next
    s = _scale_root(-(n - 1.)**2*d*c1*c2/n, b, C, s0, f)

    c_HR, c_AR = s*c1, s*c2
    d = _check_center_thickness(thickness, sagitta(c_HR, diameter),
                                sagitta(c_AR, diameter))
    got = lens_power(c_HR, c_AR, d, n)
    if np.abs(got - C) > 1e-9*np.abs(C):
        raise LensGeometryError(
            'Scaling this shape did not settle on f = %.6g m (it gives '
            '%.6g m). This is a bug in the solver, not in the lens asked '
            'for.' % (f, 1./got if got else np.inf))
    return c_HR, c_AR

def _scale_root(a, b, C, s0, f):
    '''
    Solve a*s^2 + b*s = C for the scale factor, on the branch that
    survives as the substrate thins.
    '''
    if a == 0.:
        return C/b
    disc = b*b + 4.*a*C
    if disc < 0.:
        raise LensGeometryError(
            'No lens of this shape has f = %.6g m: scaling its two faces '
            'together cannot reach that focal length from either '
            'direction. Give a shape as well, and the faces are solved '
            'for from scratch.' % (f,))
    root = np.sqrt(disc)
    lo = (-b + root)/(2.*a)
    hi = (-b - root)/(2.*a)
    return lo if np.abs(lo - s0) <= np.abs(hi - s0) else hi

#}}}

class Lens(Mirror):
    '''
    A lens: a substrate that refracts at both faces.

    Mechanically a Mirror whose two faces are both curved and both
    transmit, so everything a Mirror can do - non-sequential tracing,
    ghost beams off either face, drawing, dragging in the viewer - works
    unchanged. What it adds is the constructor: a lens is ordered by its
    focal length, and the curvatures follow.

    The focal length is not stored anywhere. ``f`` is computed from the
    curvatures, the thickness and the index whenever it is read, and
    assigning to it reshapes the faces to match. The curvatures stay the
    one description of the lens, so the two can never disagree, and
    tuning a lens is what it looks like::

        for f in np.arange(150, 400, 10)*mm:
            L.f = f
            layout.trace()
            ...

    Setting f keeps the shape - both curvatures are scaled together -
    and keeps the lens where it is. set_focal_length() takes a shape as
    well, for changing that too.

    Four defaults differ from Mirror, because they have to:

    ================  =========  ====================================
    ..                Mirror     Lens
    ================  =========  ====================================
    wedgeAngle        0.25 deg   0, or the faces are not coaxial
    HRtransmissive    False      True: the front face is meant to pass
    HRreflective      True       False: its reflections are ghosts
    Refl_HR/Trans_HR  0.99/0.01  0/1: both faces reflect nothing
    ================  =========  ====================================

    HRtransmissive matters more than it looks. With it False a beam
    passing through the front face counts as one order of stray, so the
    main beam through a lens would be a ghost, and non_seq_trace would
    stop following it at a low order. HRreflective is its mirror image:
    with it False every reflection at the front face counts one order
    of stray, as reflections at the back face already do, so the ghosts
    off a lens carry the order a ghost deserves instead of passing for
    main beams.

    Both faces default to reflecting nothing. A real lens does reflect,
    but a system of them makes so many faint ghosts that the picture is
    unreadable and the trace slow, and most of the time a lens is there
    to bend the main beam. Someone chasing the ghosts off a lens knows
    they are, and says so::

        L = Lens(f=500*mm, Refl_HR=0.005, Trans_HR=0.995,
                 Refl_AR=0.005, Trans_AR=0.995)

    Attributes
    ----------
    f : float
        Focal length. Readable and writable; see the property.
    center_thickness : float
        Distance between the two apexes, which is what a catalogue calls
        the thickness. ``thickness`` itself is Mirror's: the distance
        between the two chord planes, i.e. at the rim. Read only.

    Examples
    --------
    A 500 mm biconvex lens, 1 inch across::

        L = Lens(f=500*mm)

    The same power as a plano-convex lens with the curved face towards
    the back, put 20 cm along the axis and turned to face the beam::

        L = Lens(f=500*mm, shape='convex-plano', center=[0.2, 0.0],
                 normAngleHR=pi)

    A diverging lens. The symmetric default follows the sign of f, so
    this is biconcave::

        L = Lens(f=-100*mm, thickness=3*mm)

    A meniscus, which f alone does not determine: one radius is given
    and the other is solved for. ROC_HR is positive for a concave front
    face, as everywhere in gtrace::

        L = Lens(f=200*mm, shape='meniscus', ROC_HR=-50*mm)
    '''

    #A lens is placed by its middle: the beam goes through it, so there
    #is no reflection point for the faces to stay under, and it is the
    #substrate that sits at a position on the bench. Changing a
    #curvature therefore moves the faces and leaves the lens where it
    #is - the opposite of a mirror, and the reason anchor_point exists.
    anchor_point = Enum(['center', 'HRcenter'])

    #A lens has no reflective side to mark.
    draw_HR_marker = False

#{{{ __init__

    def __init__(self, f=None, shape=None,
                 center=None, HRcenter=None, normAngleHR=0.0,
                 normVectHR=None, diameter=1*inch, thickness=6*mm,
                 n=1.45, ROC_HR=None, inv_ROC_HR=None, inv_ROC_AR=None,
                 wedgeAngle=0.0, Refl_HR=0.0, Trans_HR=1.0,
                 Refl_AR=0.0, Trans_AR=1.0, name="Lens",
                 HRtransmissive=True, HRreflective=False,
                 term_on_HR=False, term_on_HR_order=0,
                 term_on_HR_transmits=False,
                 max_stray_order=None):
        '''
        Create a lens.

        Parameters
        ----------
        f : float or None, optional
            Focal length. Positive converges. The curvatures are solved
            for from it. None instead takes the curvatures directly
            through inv_ROC_HR and inv_ROC_AR, which is how copy() and
            the layout loader rebuild a lens whose radii may since have
            been edited.
        shape : str or None, optional
            Which shape of that focal length, spelt HR face first:
            'biconvex', 'biconcave', 'plano-convex', 'convex-plano',
            'plano-concave', 'concave-plano', 'convex-concave',
            'concave-convex', or the family names 'symmetric' and
            'meniscus'. None means symmetric, which follows the sign of
            f. Only meaningful together with f.
        center : array or None, optional
            Position of the centre of the substrate. A lens is normally
            placed by its middle rather than by a face. Defaults
            [0.0, 0.0] when HRcenter is not given either.
        HRcenter : array or None, optional
            Position of the apex of the front face, as for a Mirror.
            Mutually exclusive with center.
        normAngleHR : float, optional
            Direction angle of the normal of the front face, in radians.
            Defaults 0.0.
        normVectHR : array or None, optional
            The same as a vector. Defaults None.
        diameter : float, optional
            Aperture. Defaults 1 inch.
        thickness : float, optional
            Distance between the two chord planes: the thickness at the
            rim, and Mirror's thickness. Defaults 6 mm. For a concave
            lens this is the thick part, and it has to be enough to
            leave the middle standing.
        n : float, optional
            Index of refraction. Defaults 1.45.
        ROC_HR : float or None, optional
            Radius of the front face, for a meniscus. Positive for a
            concave face. Mutually exclusive with inv_ROC_HR.
        inv_ROC_HR, inv_ROC_AR : float or None, optional
            The curvatures themselves, for building a lens without
            solving for one. inv_ROC_HR doubles as the given face of a
            meniscus.
        wedgeAngle : float, optional
            Wedge between the faces, in radians. Defaults 0: a lens
            with a wedge is a prism as well as a lens, and the focal
            length solved for here assumes coaxial faces.
        Refl_HR, Trans_HR, Refl_AR, Trans_AR : float, optional
            Power reflectivity and transmissivity of the two faces.
            Default to reflecting nothing, so that a lens makes no
            ghosts. Give the faces a real coating - 0.5% is typical -
            when the ghosts are what you are after. See the class
            docstring.
        name : str, optional
            Defaults "Lens".
        HRtransmissive : boolean, optional
            Defaults True, unlike Mirror. See the class docstring.
        HRreflective : boolean, optional
            Defaults False, unlike Mirror. See the class docstring.
        term_on_HR : boolean, optional
            Defaults False.
        term_on_HR_order : int, optional
            Defaults 0.
        term_on_HR_transmits : boolean, optional
            Defaults False.
        max_stray_order : int or None, optional
            Defaults None.

        Raises
        ------
        LensGeometryError
            If the lens cannot be made out of the blank given, or if the
            arguments over- or under-determine it.
        '''
        if ROC_HR is not None:
            if inv_ROC_HR is not None:
                raise LensGeometryError(
                    'Give ROC_HR or inv_ROC_HR, not both.')
            if ROC_HR == 0:
                raise LensGeometryError(
                    'ROC_HR cannot be zero. A flat face is inv_ROC_HR=0, '
                    'or one of the plano shapes.')
            inv_ROC_HR = 1./ROC_HR

        if center is not None and HRcenter is not None:
            raise LensGeometryError(
                'Give center or HRcenter, not both: one is the middle of '
                'the substrate, the other the apex of its front face.')

        if f is None:
            if shape is not None:
                raise LensGeometryError(
                    'shape=%r has no meaning without f. With no focal '
                    'length to solve for, the faces are whatever '
                    'inv_ROC_HR and inv_ROC_AR say.' % (shape,))
            c_HR = 0.0 if inv_ROC_HR is None else float(inv_ROC_HR)
            c_AR = 0.0 if inv_ROC_AR is None else float(inv_ROC_AR)
        else:
            if inv_ROC_AR is not None:
                raise LensGeometryError(
                    'inv_ROC_AR is solved for from f. Giving both would '
                    'over-determine the lens.')
            #Which shapes take a radius, and which insist on one, is the
            #solver's business; it raises for both.
            c_HR, c_AR = solve_lens_curvatures(
                f, shape=shape, diameter=diameter, thickness=thickness,
                n=n, inv_ROC_HR=inv_ROC_HR)

        #The blank has to survive the faces cut into it. The solver
        #checks this too; repeating it here covers the f=None path,
        #where a lens is rebuilt from curvatures that may have been
        #edited into something impossible since.
        _check_center_thickness(thickness, sagitta(c_HR, diameter),
                                sagitta(c_AR, diameter))

        Mirror.__init__(self,
                        HRcenter=[0.0, 0.0] if HRcenter is None else HRcenter,
                        normAngleHR=normAngleHR, normVectHR=normVectHR,
                        diameter=diameter, thickness=thickness,
                        wedgeAngle=wedgeAngle, inv_ROC_HR=c_HR,
                        inv_ROC_AR=c_AR, Refl_HR=Refl_HR,
                        Trans_HR=Trans_HR, Refl_AR=Refl_AR,
                        Trans_AR=Trans_AR, n=n, name=name,
                        HRtransmissive=HRtransmissive,
                        HRreflective=HRreflective,
                        term_on_HR=term_on_HR,
                        term_on_HR_order=term_on_HR_order,
                        term_on_HR_transmits=term_on_HR_transmits,
                        max_stray_order=max_stray_order)

        #Placing by the middle is the natural thing for a lens, and it
        #is also the default: HRcenter=[0,0] above is only a stand-in
        #for the constructor, replaced here unless it was asked for.
        if HRcenter is None:
            self.center = [0.0, 0.0] if center is None else center

#}}}

#{{{ copy

    def copy(self):
        #From the curvatures, not from f: a lens whose radii were edited
        #after it was built is that lens, not the one originally
        #ordered, and re-solving would quietly reshape it.
        m = Lens(inv_ROC_HR=self.inv_ROC_HR, inv_ROC_AR=self.inv_ROC_AR,
                    HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                    diameter=self.diameter, thickness=self.thickness,
                    wedgeAngle=self.wedgeAngle, Refl_HR=self.Refl_HR,
                    Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR,
                    Trans_AR=self.Trans_AR, n=self.n, name=self.name,
                    HRtransmissive=self.HRtransmissive,
                    HRreflective=self.HRreflective,
                    term_on_HR=self.term_on_HR,
                    term_on_HR_order=self.term_on_HR_order,
                    term_on_HR_transmits=self.term_on_HR_transmits,
                    max_stray_order=self.max_stray_order)
        m.anchor_point = self.anchor_point
        return m

#}}}

#{{{ Derived quantities

    @property
    def center_thickness(self):
        '''
        Distance between the two apexes: the thickness a catalogue
        quotes. thickness is the distance between the chord planes,
        i.e. at the rim.
        '''
        return self.thickness + self.sagHR + self.sagAR

    @property
    def f(self):
        '''
        Focal length of the lens.

        Reading it computes it from the curvatures, the centre thickness
        and the index as they stand, so it cannot go stale. Infinite for
        a substrate with no power left in it.

        Assigning to it reshapes the two faces until the lens has that
        focal length, keeping its shape and leaving it where it is::

            for f in np.arange(150, 400, 10)*mm:
                L.f = f
                layout.trace()

        See set_focal_length(), which this calls, for what "keeping its
        shape" means and for how to change the shape as well.
        '''
        P = lens_power(self.inv_ROC_HR, self.inv_ROC_AR,
                       self.center_thickness, self.n)
        return np.inf if P == 0. else 1./P

    @f.setter
    def f(self, value):
        self.set_focal_length(value)

    def set_focal_length(self, f, shape=None, ROC_HR=None):
        '''
        Give the lens a new focal length.

        Parameters
        ----------
        f : float
            The focal length wanted.
        shape : str or None, optional
            None, the usual case, keeps the shape the lens already has:
            both curvatures are scaled by one common factor, so a flat
            face stays flat and a meniscus keeps its bend. Asking a
            converging lens for a negative focal length turns it inside
            out, and the shape name follows.

            A name from the list the constructor takes instead solves
            both faces from scratch, which is how a lens changes shape
            without being rebuilt and losing its place in the layout.
        ROC_HR : float or None, optional
            The front radius, when shape is 'meniscus'.

        Raises
        ------
        LensGeometryError
            If the new focal length cannot be had. The lens is left
            exactly as it was: the whole solve, and every check on it,
            happens before anything is assigned.

        Notes
        -----
        Whatever anchor_point names stays put, which for a lens is the
        centre of the substrate. The faces move, since the sagittae
        change with the curvature, but the lens as a whole does not
        wander up the bench as it is tuned.
        '''
        if shape is None:
            c_HR, c_AR = rescale_lens_curvatures(
                self.inv_ROC_HR, self.inv_ROC_AR, f, diameter=self.diameter,
                thickness=self.thickness, n=self.n)
            if ROC_HR is not None:
                raise LensGeometryError(
                    'ROC_HR pins the front face of a meniscus solved from '
                    'scratch. Without a shape the faces keep the ratio '
                    'they already have, and there is nothing to pin.')
        else:
            c_HR, c_AR = solve_lens_curvatures(
                f, shape=shape, diameter=self.diameter,
                thickness=self.thickness, n=self.n,
                inv_ROC_HR=None if ROC_HR is None else 1./ROC_HR)

        #Everything above can raise, and none of it has touched the
        #lens. From here on nothing can: anchor_point holds the substrate
        #still while the faces move on it.
        self.inv_ROC_HR = c_HR
        self.inv_ROC_AR = c_AR

    @property
    def shape(self):
        '''
        The name of the shape this lens currently has, spelt HR face
        first, or 'plano-plano' for a substrate with no power.
        '''
        faces = (_face_kind(self.inv_ROC_HR), _face_kind(self.inv_ROC_AR))
        for name, (family, want_HR, want_AR) in sorted(_LENS_SHAPES.items()):
            if (want_HR, want_AR) == faces:
                return name
        return '%s-%s' % faces

#}}}

#}}}

class CyLens(Lens, CyMirror):
    '''
    A cylindrical lens: a substrate that refracts at both faces and
    focuses in one plane only.

    It is ordered exactly like a Lens - by its focal length, solved as a
    thick lens - and shaped exactly like a CyMirror: both faces are
    cylinders sharing one curve_direction, so the focal length lives in
    that plane and the other plane sees nothing but a window. The two
    parents each contribute what they already know. Lens brings the
    ordering - the f property, set_focal_length(), shape, and the anchor
    on the middle of the substrate - and CyMirror brings the geometry
    and the ray matrices that put the power in one plane only.

    The solver needs no change for a cylinder. In the curved plane a
    cylindrical lens is the lensmaker's thick lens, and the distance
    between the apexes is ``thickness + sagHR + sagAR`` for either
    curve_direction: the axis of a cylinder curved out of the drawing
    lies in it, so the section the trace sees runs through the apexes.

    As with CyMirror, only 'h' is visible in the drawing. A 'v' lens is
    drawn as a plain rectangle - what the plane of the trace cuts out of
    it - and its focusing happens entirely out of the page, carried by
    the beam's qy.

    The focal length f is quoted at normal incidence. Tilted, the two
    planes scale differently - see CyMirror and cyl_refl_defl_angle for
    what a tilt does to each plane.

    Examples
    --------
    A 500 mm cylindrical lens focusing in the plane of the drawing::

        L = CyLens(f=500*mm)

    The same power focusing out of the plane, which is drawn straight::

        L = CyLens(f=500*mm, curve_direction='v')

    The shapes a Lens can take, a CyLens can too::

        L = CyLens(f=-100*mm, thickness=3*mm, shape='plano-concave')
    '''

#{{{ __init__

    def __init__(self, f=None, shape=None,
                 center=None, HRcenter=None, normAngleHR=0.0,
                 normVectHR=None, diameter=1*inch, thickness=6*mm,
                 n=1.45, ROC_HR=None, inv_ROC_HR=None, inv_ROC_AR=None,
                 wedgeAngle=0.0, Refl_HR=0.0, Trans_HR=1.0,
                 Refl_AR=0.0, Trans_AR=1.0, name="CyLens",
                 HRtransmissive=True, HRreflective=False,
                 term_on_HR=False, term_on_HR_order=0,
                 term_on_HR_transmits=False,
                 max_stray_order=None, curve_direction='h'):
        '''
        Create a cylindrical lens.

        Takes everything Lens.__init__ takes, with the same meanings
        and the same defaults, plus:

        Parameters
        ----------
        curve_direction : str, optional
            Which plane the faces curve in, and so which plane the
            focal length lives in. 'h' is the plane of the drawing,
            'v' is perpendicular to it. Defaults 'h'.

        Raises
        ------
        LensGeometryError
            If the lens cannot be made out of the blank given, if the
            arguments over- or under-determine it, or if
            curve_direction is neither 'h' nor 'v'.
        '''
        #CyMirror stores whatever it is given and branches on it face
        #by face, so a typo would be half one direction and half the
        #other. A lens ordered from a catalogue can afford to check.
        if curve_direction not in ('h', 'v'):
            raise LensGeometryError(
                "curve_direction must be 'h' or 'v', not %r."
                % (curve_direction,))

        Lens.__init__(self, f=f, shape=shape, center=center,
                      HRcenter=HRcenter, normAngleHR=normAngleHR,
                      normVectHR=normVectHR, diameter=diameter,
                      thickness=thickness, n=n, ROC_HR=ROC_HR,
                      inv_ROC_HR=inv_ROC_HR, inv_ROC_AR=inv_ROC_AR,
                      wedgeAngle=wedgeAngle, Refl_HR=Refl_HR,
                      Trans_HR=Trans_HR, Refl_AR=Refl_AR,
                      Trans_AR=Trans_AR, name=name,
                      HRtransmissive=HRtransmissive,
                      HRreflective=HRreflective,
                      term_on_HR=term_on_HR,
                      term_on_HR_order=term_on_HR_order,
                      term_on_HR_transmits=term_on_HR_transmits,
                      max_stray_order=max_stray_order)

        #After Lens.__init__, like CyMirror sets it after the Mirror
        #skeleton is up: nothing the constructor runs reads it, and the
        #methods that branch on it are not called until the lens is
        #whole.
        self.curve_direction = curve_direction

#}}}

#{{{ copy

    def copy(self):
        #From the curvatures, not from f, for the same reason as
        #Lens.copy().
        m = CyLens(inv_ROC_HR=self.inv_ROC_HR, inv_ROC_AR=self.inv_ROC_AR,
                   HRcenter=self.HRcenter, normAngleHR=self.normAngleHR,
                   diameter=self.diameter, thickness=self.thickness,
                   wedgeAngle=self.wedgeAngle, Refl_HR=self.Refl_HR,
                   Trans_HR=self.Trans_HR, Refl_AR=self.Refl_AR,
                   Trans_AR=self.Trans_AR, n=self.n, name=self.name,
                   HRtransmissive=self.HRtransmissive,
                   HRreflective=self.HRreflective,
                   term_on_HR=self.term_on_HR,
                   term_on_HR_order=self.term_on_HR_order,
                   term_on_HR_transmits=self.term_on_HR_transmits,
                   max_stray_order=self.max_stray_order,
                   curve_direction=self.curve_direction)
        m.anchor_point = self.anchor_point
        return m

#}}}
