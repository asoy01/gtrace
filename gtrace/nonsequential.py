'''
gtrace.nonsequential

A module to perform non-sequential trace of a beam
in an optical system.
'''

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

#{{{ non_seq_trace

def non_seq_trace(optList, src_beam, order=10, power_threshold=0.1, open_beam_length=1.0):
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
        An integer to specify how many times the internal reflections
        are computed.
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
        #The beam does not hit any optics
        return [src_beam]

    if final_answer['face'] == 'side':
        #The beam is terminated on a side of an optics
        src_beam.length = final_answer['distance']
        return [src_beam]

    #The beam hits an actual optical surface

    #Avoid forming a cavity
    if hit_optics.term_on_HR  and final_answer['face'] == 'HR' and \
                src_beam.stray_order <= hit_optics.term_on_HR_order:

        src_beam.length = final_answer['distance']
        return [src_beam]

    ans = hit_optics.hit(src_beam, order=order, threshold=power_threshold,
                         face=final_answer['face'])

    terminated_beam_list = [b for b in list(ans[1].values()) if b.incSurfAngle is not None]
    open_beam_list = [b for b in list(ans[1].values()) if b.incSurfAngle is None]

    #For each open beam, perform the non sequential trace
    for b in open_beam_list:
        b.length = open_beam_length
        beams = non_seq_trace(optList=optList, src_beam=b.copy(), order=order,
                              power_threshold=power_threshold)
        terminated_beam_list.extend(beams)

    return terminated_beam_list

#}}}
