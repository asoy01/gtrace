'''
gtrace.nonsequential

A module to perform non-sequential trace of a beam
in an optical system.
'''

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
__version__ = "0.7.0"
__maintainer__ = "Yoichi Aso"
__email__ = "asoy01@gmail.com"
__status__ = "Beta"

#}}}

#{{{ non_seq_trace

def non_seq_trace(optList, src_beam, order=10, power_threshold=0.1,
                  open_beam_length=1.0):
    '''
    Perform non-sequential trace of the source beam, src_beam,
    through the optical system represented by a collection of optics,
    optList.

    Parameters
    ----------
    optList: list of gtrace.optcomp.Optics
        List of optical components.
    src_beam: gtrace.beam.GaussianBeam
        The source beam object.
    order: int, optional
        Number of ghost reflections a beam may go through before it
        stops being followed. Every beam carries the count as its
        stray_order, and this function does not reset it when the beam
        leaves one element for the next, so the limit applies to the
        whole trace rather than to one element. An optics whose
        max_stray_order is set overrides this for itself, since how
        deep its ghosts are worth chasing is a property of the element
        rather than of the trace.
        Defaults to 10.
    power_threshold: float, optional
        The power threshold for internal reflection calculation.
        If the power of an auxiliary beam falls below this threshold,
        further propagation of this beam will not be performed.
        Defaults to 0.1.
    open_beam_length: float, optional
        The default length for beams that are not hitting anything.
        Defaults to 1.0.

    Returns
    -------
    terminated_beam_list: list of gtrace.beam.GaussianBeam
        A list of beams.
    '''
    #Loop over all the optics to see if the source beam hit them.
    #Then select the closest optics being hit.
    min_dist = 1e15
    final_answer = None
    hit_optics = None
    for opt in optList:
        #See if the beam hit the optics
        ans = opt.isHit(src_beam)
        #If the beam hit the optics
        if ans['isHit']:
            #If the intersection point is closest one so far
            if min_dist > ans['distance']:
                    min_dist = ans['distance']
                    final_answer = ans
                    hit_optics = opt

    if final_answer is None:
        #The beam does not hit any optics. Its length is not measured
        #from anything: it is open_beam_length, or the length the
        #source was given. open_end says so, since nothing else about
        #the beam does, and a drawing that lengthens such a beam has
        #to know which ones it may lengthen. A beam that ends on a
        #surface or on a side is not one of them - its length is where
        #the glass is.
        src_beam.open_end = True
        return [src_beam]

    if final_answer['face'] == 'side':
        #The beam is terminated on a side of an optics
        src_beam.length = final_answer['distance']
        return [src_beam]

    #The beam hits an actual optical surface

    #Avoid forming a cavity.
    #
    #What is stopped is the reflection: two facing high reflectors pass
    #the main beam between them until the trace gives up, and the beam
    #that does that is the external reflection off the HR.
    #
    #term_on_HR_transmits says whether stopping it means stopping
    #everything. False, the default and what this has always done, ends
    #the beam at the surface and computes nothing. True lets the element
    #be hit as usual and drops the one beam that would come back, so a
    #mirror can be a cavity mirror and still be looked through - the
    #beam transmitted out of the far side is often the one a detector
    #sees. Everything that survives is counted and capped as it would be
    #anywhere else, since it goes through hit() by the ordinary route.
    drop_reflection = False
    if hit_optics.term_on_HR  and final_answer['face'] == 'HR' and \
                src_beam.stray_order <= hit_optics.term_on_HR_order:

        if not getattr(hit_optics, 'term_on_HR_transmits', False):
            src_beam.length = final_answer['distance']
            return [src_beam]

        drop_reflection = True

    #An optics may cap the stray order it is worth computing for itself.
    hit_order = getattr(hit_optics, 'max_stray_order', None)
    if hit_order is None:
        hit_order = order

    ans = hit_optics.hit(src_beam, order=hit_order, threshold=power_threshold,
                         face=final_answer['face'])

    produced = ans[1]
    if drop_reflection:
        #'r1' is the external reflection, and the only beam here that
        #can return to the element the source came from at the power a
        #cavity needs. The ghosts that leave through the HR from inside
        #the substrate ('r2' onwards) are not dropped: each is a round
        #trip weaker and costs a stray order, so order and the power
        #threshold end them, and they are what a ghost hunt is looking
        #for. hit() is left alone - it is the sequential interface, and
        #code that calls hitFromHR directly asks for 'r1' by name.
        produced = dict(produced)
        produced.pop('r1', None)

    terminated_beam_list = [b for b in list(produced.values()) if b.incSurfAngle is not None]
    open_beam_list = [b for b in list(produced.values()) if b.incSurfAngle is None]

    #For each open beam, carry on through the rest of the system.
    #The stray order rides with the beam: what it has already cost
    #to make is what it costs, and `order` is the budget for the
    #whole trace rather than a fresh allowance at every element.
    #Zeroing it here - which this did until 2026-08 - made every
    #element hand out `order` ghosts again, so nothing bounded the
    #recursion but the power threshold, and a ghost arriving
    #somewhere new was drawn as a main beam.
    for b in open_beam_list:
        b.length = open_beam_length
        beams = non_seq_trace(optList=optList, src_beam=b.copy(), order=order,
                              power_threshold=power_threshold,
                              open_beam_length=open_beam_length)
        terminated_beam_list.extend(beams)

    return terminated_beam_list

#}}}
