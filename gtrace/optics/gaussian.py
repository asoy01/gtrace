"""
gaussian - Gaussian Optics Module

This module contains several utility functions for gaussian optics.
"""

#{{{ Import modules
import numpy as np
pi = np.pi
sqrt = np.lib.scimath.sqrt
from .unit import *
import scipy.special as spf

#}}}

#{{{ modeSpacing

def modeSpacing(g1, g2):
    return np.arccos(np.sqrt(g1*g2))/pi

#}}}

#{{{ q-parameter related functions

def q2zr(q):
    '''
    Convert a q-parameter to Rayleigh range.

    Parameters
    ----------
    q : complex
        Beam parameter.

    Returns
    -------
    zr : float
        Rayleigh range.
    '''
    zr = np.float(np.imag(q))
    return zr

def q2w(q, wl=1064*nm):
    '''
    Convert a q-parameter to the beam size

    Parameters
    ----------
    q : complex
        Beam parameter.
    wl : float, optional
        Wavelength.
        Defaults 1064*nm.

    Returns
    -------
    w : float
        Beam size.
    '''
    S = -1.0/np.imag(1.0/q)
    w = np.sqrt(wl*S/pi)
    return w


def q2R(q):
    '''
    Convert a q-parameter to the ROC

    Parameters
    ----------
    q : complex
        Beam parameter.

    Returns
    -------
    float
        Radius of curvature.
    '''
    return 1.0/np.real(1.0/q)

def Rw2q(ROC=1.0, w=1.0, wl=1064e-9):
    '''
    Get the q-parameter from the ROC and w.

    Paramters
    ---------
    ROC : float, optional
        Radius of curvature.
    w : float, optional
        Beam size.
        Defaults 1.0.
    wl : float, optional
        Wavelength.
        Defaults 1064*nm.

    Returns
    -------
    complex
        Beam parameter.
    '''
    k = 2.0*pi/wl
    S = w**2 * k/2

    return 1.0/(1.0/ROC + 1.0/(1j*S))


def InvROCandW2q(invROC=0.0, w=1.0, wl=1064e-9):
    '''
    Get the q-parameter from the inverse ROC and w.

    Parameters
    ----------
    invROC : float, optional
        Inverse of the ROC.
        Defaults 0.0.
    w : float, optional
        Beam size.
        Defaults 1.0.
    wl : float, optional
        Wavelength.
        Defaults 1064*nm.

    Returns
    -------
    complex
        Beam parameter.
    '''
    k = 2.0*pi/wl
    S = w**2 * k/2

    return 1.0/(invROC + 1.0/(1j*S))

def zr2w0(zr, wl=1064*nm):
    '''
    Convert Rayleigh range to the waist size

    Parameters
    ----------
    zr : float
        Rayleigh range.
    wl : float, optional
        Wavelength.
        Defaults 1064*nm.

    Returns
    -------
    float
        Waist size.
    '''
    return np.sqrt(2*zr*wl/(2*pi))

def w02zr(w0, wl=1064*nm):
    '''
    Convert waist size to Rayleigh range

    Parameters
    ----------
    w0 : float
        Waist size.
    wl : float, optional
        Wavelength.
        Defaults 1064*nm.
    '''
    return (2*pi/wl)*(w0**2)/2

def modeMatching(q1, q2x, q2y=False):
    '''
    Mode matching between two beams with different q-parameters.
    The axes of the two beams are assumed to be matched.

    Parameters
    ----------
    q1 : complex
        q-parameter of the first beam. This beam is assumed to be circular.

    q2x : complex
        q-parameter of the second beam in x-direction. If the second beam
        is also circular, omit the next argument.

    q2y : complex, optional
        q-parameter of the second beam in y-direction. Specify this parameter
        if the second beam is eliptic.
        Defaults False.
    '''

    zr1 = np.imag(q1)
    d1 = np.real(q1)

    if q2y:
        #Eliptic beam
        zrx = np.imag(q2x)
        dx = np.real(q2x)
        zry = np.imag(q2y)
        dy = np.real(q2y)
        return np.abs(2*sqrt(zr1*sqrt(zrx*zry)/((zr1+zrx+1j*(dx-d1))*(zr1+zry+1j*(dy-d1)))))**2
    else:
        #Circular beam
        zr2 = np.imag(q2x)
        d2 = np.real(q2x)
        ec = zr2 - zr1
        az = d2 -d1

        return np.abs(2*zr1*sqrt(1+ec/zr1)/(2*zr1+ec+1j*az))**2

def modeMatchingElliptic(q1x, q1y, q2x, q2y):
    '''
    Mode matching between two elliptic beams.

    Parameters
    ----------
    q1x : complex
        q-parameter of the first beam in x-direction.
    q1y : complex
        q-parameter of the first beam in y-direction.
    q2x : complex
        q-parameter of the second beam in x-direction.
    q2y : complex
        q-parameter of the second beam in y-direction.

    Returns
    -------
    float
    '''

    zr1x = np.imag(q1x)
    d1x = np.real(q1x)
    zr1y = np.imag(q1y)
    d1y = np.real(q1y)

    zr2x = np.imag(q2x)
    d2x = np.real(q2x)
    zr2y = np.imag(q2y)
    d2y = np.real(q2y)
    return np.abs(2*sqrt(sqrt(zr1x*zr1y*zr2x*zr2y)/((zr1x+zr2x+1j*(d2x-d1x))*(zr1y+zr2y+1j*(d2y-d1y)))))**2


def optimalMatching(q1, q2):
    '''
    Returns a mode (q-parameter) which best matches the given
    two q-parameters, q1 and q2.

    Parameters
    ----------
    q1 : complex
        q-parameter of the first beam. This beam is assumed to be circular.
    q2 : complex
        q-parameter of the second beam. This beam is assumed to be circular.

    Returns
    -------
    (complex, match?)
        (q, match)

        q: The best matching q-parameter

        match: Mode matching rate
    '''

    zr1 = np.imag(q1)
    d1 = np.real(q1)

    zr2 = np.imag(q2)
    d2 = np.real(q2)

    zr = np.sqrt(zr1*zr2)*np.sqrt(1+((d2-d1)/(zr1+zr2))**2)
    d = (zr1*d2 + zr2*d1)/(zr1+zr2)

    q = d +1j*zr
    match = modeMatching(q, q1, q2)

    return (q, match)

#{{{ For compatibility

def qToRadius(q, wl=1064e-9):
    """
    Convert a q-parameter to the beam size

    Parameters
    ----------
    q : complex
        Beam parameter.
    wl : float, optional
        Wavelength.
        Defaults 1064e-9.

    Returns
    -------
    float
        Radius.
    """
    k = 2*pi/wl
    return sqrt(-2.0/(k*np.imag(1.0/q)))

def qToROC(q):
    """Convert a q-parameter to radius of curvature.

    Parameters
    ----------
    q : complex
        Beam parameter.

    Returns
    -------
    float
        Radius of curvature.
    """
    return 1.0/np.real(1.0/q)

def ROCandWtoQ(ROC=1.0, w=1.0, wl=1064e-9):
    """Convert radius of curvature and beam width to q-parameter

    Parameters
    ----------
    ROC: float, optional
        Radius of curvature.
        Defaults to 1.0.
    w: float, optional
        Beam width.
        Defaults to 1.0.
    wl: float, optional
        Wavelength.
        Defaults to 1064e-9

    Returns
    -------
    complex
        q-parameter
    """
    k = 2.0*pi/wl
    S = w**2 * k/2

    return 1.0/(1.0/ROC + 1.0/(1j*S))

#}}}

#}}}

#{{{ Beam clip

def beamClip(a = 1.0,  w = 3.0):
    """Beam clip

    Parameters
    ----------
    a : float
    w : float

    Returns
    -------
    float
    """
    return (1+spf.erf(np.sqrt(2)*a/w))/2

def appertureCut(r = 1.0,  w = 3.0):
    """Apperture cut.

    Parameters
    ----------
    r : float
    w : float

    Returns
    -------
    float
    """
    return 1-np.exp(-2*(r/w)**2)

#}}}
