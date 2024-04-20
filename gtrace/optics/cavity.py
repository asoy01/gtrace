"""
cavity.py - A Cavity class and related functions for representing a Fabry-Perot cavity
"""

#{{{ Import modules
import numpy as np
pi = np.pi
c = 299792458.0
sqrt = np.lib.scimath.sqrt
import gtrace.optics.gaussian as gauss
from traits.api import CFloat, HasTraits
#}}}

#{{{ Class Cavity

class Cavity(HasTraits):
    """
    A class to represent a Fabry-Perot cavity.

    Attributes
    ----------
    r1 : float
        Input mirror reflectivity (amplitude)
    r2 : float
        End mirror reflectivity (amplitude)
    rp1 : float
        Input mirror reflectivity (power)
    rp2 : float
        End mirror reflectivity (power)
    L : float
        Length
    R1 : float
        ROC of the input mirror (positive when concave to incident light, i.e.
        convex seen from inside the cavity)
    R2 : float
        ROC of the end mirror (positive when concave to incident light, i.e.
        concave seen from inside the cavity)
    wl : float
        Wavelength
    """
    r1 = CFloat(0.9)
    r2 = CFloat(0.99)
    rp1 = CFloat()
    rp2 = CFloat()
    tp1 = CFloat()
    tp2 = CFloat()
    t1 = CFloat()
    t2 = CFloat()

    L = CFloat(1.0)
    R1 = CFloat(-1.5)
    R2 = CFloat(1.5)
    g1 = CFloat()
    g2 = CFloat()
    wl=CFloat()

    def __init__(self, r1=0.9, r2=0.99, L=1.0, R1=-1.5, R2=1.5, wl=1064e-9, power=False):
        """
        Parameters
        ----------
        r1 : float, optional
            Input mirror reflectivity.
            Defaults 0.9.
        r2 : float, optional
            End mirror reflectivity.
            Defaults 0.99.
        L : float, optional.
            Length.
            Defaults 1.0.
        R1 : float, optional
            ROC of the input mirror (positive when concave to incident light, i.e.
            convex seen from inside the cavity)
            Defaults -1.5.
        R2 : float, optional
            ROC of the end mirror (positive when concave to incident light, i.e.
            concave seen from inside the cavity).
            Default 1.5.
        wl : float, optional
            Wavelength.
            Defaults 1064e-9.
        power : boolean, optional
            If True, r1 and r2 are treated as power reflectivities.
            Otherwise, r1 and r2 are regarded as amplitude reflectivities.
            Defaults False.
        """
        if power:
            self.r1 = np.sqrt(r1)
            self.r2 = np.sqrt(r2)
        else:
            self.r1 = r1
            self.r2 = r2

        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.wl = wl

    #{{{ Trait change handlers
    def _R1_changed(self, old, new):
        self.g1 = 1 + self.L/new

    def _R2_changed(self, old, new):
        self.g2 = 1 - self.L/new

    def _L_changed(self, old, new):
        self.g1 = 1 + new/self.R1
        self.g2 = 1 - new/self.R2

    def _r1_changed(self, old, r1):
        self.rp1 = r1**2
        self.tp1 = 1 - self.rp1
        self.t1 = np.sqrt(self.tp1)

    def _r2_changed(self, old, r2):
        self.rp2 = r2**2
        self.tp2 = 1 - self.rp2
        self.t2 = np.sqrt(self.tp2)

    #}}}

    def finesse(self):
        '''
        Returns the finesse of the cavity.
        '''
        return finesse(self.r1, self.r2)

    def storageTime(self):
        '''
        Storage time
        '''

        return 2*self.L*self.finesse()/(2*pi*c)

    def pole(self):
        '''
        Cavity pole frequency [Hz]
        '''

        return 1/(2*pi*self.storageTime())

    def Nbounce(self):
        '''
        Bounce number
        '''

        return 2*self.finesse()/pi

    def powerGain(self):
        '''
        Ratio of the intra-cavity power to the input power.
        '''
        return (self.t1/(1-self.r1*self.r2))**2

    def FSR(self):
        '''
        Returns the free spectral range of the cavity.
        '''
        return c/(2*self.L)

    def modeSpacing(self):
        '''
        Return the transverse mode spacing of the cavity (commonly called gamma).
        It is a fractional number defined by gamma = (mode spacing frequency)/FSR.
        '''
        return gauss.modeSpacing(self.g1, self.g2)

    def waist(self, size=False):
        """
        Return the q-parameter or the radius of the beam at the cavity waist.

        Parameters
        ----------
        size : boolean, optional
            if set to true, the first element of the returned tuple will be the waist size, rather than the q-parameter.

        Returns
        -------
        (q0, d) : (complex, float)
            This function returns a tuple with two elements.
            The first element is the q-parameter of the cavity mode at
            the cavity waist. If size=True is given, it becomes the waist
            size (1/e^2 radius).
            The second element is the distance of the cavity waist from
            the input mirror.
        """
        # q0 = 1j*np.sqrt(self.L)*np.sqrt(-(self.L+self.R1)*(self.L-self.R2)\
        #                                 *(self.L+self.R1-self.R2))\
        #                                 /(2*self.L+self.R1-self.R2)

        q0 = 1j*sqrt(-self.L*(self.L+self.R1)**2*(self.L-self.R2)*\
                         (self.L+self.R1-self.R2)/((-2*self.L-self.R1+self.R2)**2))\
                         /sqrt(self.L+self.R1)

        d = self.L*(self.L-self.R2)/(2*self.L+self.R1-self.R2)

        if size:
            return (gauss.q2w(q0, wl=self.wl), d)
        else:
            return (q0, d)

    def spotSize(self):
        """
        Returns the beam spot sizes on the input and end mirrors
        as a tuple (w1,w2).
        """
        (q0,d) = self.waist()
        w1=gauss.q2w(q0-d, wl=self.wl)
        w2=gauss.q2w(q0+self.L-d, wl=self.wl)
        return (w1, w2)

    def trans(self, f=0, d=0):
        """
        Returns the amplitude transmissivity of the cavity.
        It assumes the cavity was locked to the incident light first. Then computes the
        amplitude transmissivity for the light with a frequency shift f from the original light
        with the cavity length changed by d from the initial state.

        Parameters
        ----------
        f : float, optional
            Frequency shift of the light in Hz.
            Defaults 0.
        d : float, optional.
            Cavity length detuning in m.
            Defaults 0.

        Returns
        -------
        complex
            The amplitude transmissivity of the cavity (a complex number).

        """
        #One way phase change
        phi=2*pi*(self.L*f/c + d/self.wl + d*f/c)

        return self.t1 * self.t2 * np.exp(-1j*phi)/\
                   (1-self.r1*self.r2*np.exp(-2j*phi))

    def refl(self, f=0, d=0):
        """
        Returns the amplitude reflectivity of the cavity.
        It assumes the cavity was locked to the incident light first. Then computes the
        amplitude reflectivity for the light with a frequency shift f from the original light
        with the cavity length changed by d from the initial state.

        Parameters
        ----------
        f : float, optional
            Frequency shift of the light in Hz.
            Defaults 0.
        d : float, optional.
            Cavity length detuning in m.
            Defaults 0.

        Returns
        -------
        complex
            The amplitude reflectivity of the cavity (a complex number).
        """

        #One way phase change
        phi=2*pi*(self.L*f/c + d/self.wl + d*f/c)

        return -self.r1+self.t1**2 * self.r2 *np.exp(-2j*phi)/\
                   (1-self.r1*self.r2*np.exp(-2j*phi))

    def intra(self, f=0, d=0):
        """
        Returns the intra cavity field amplitude.
        It assumes the cavity was locked to the incident light first. Then computes the
        intra-cavity field amplitude for the light with a frequency shift f from the original light
        with the cavity length changed by d from the initial state.

        Parameters
        ----------
        f : float, optional
            Frequency shift of the light in Hz.
            Defaults 0.
        d : float, optional.
            Cavity length detuning in m.
            Defaults 0.

        Returns
        -------
        complex
            The intra-cavity field amplitude at the input mirror surface (a complex number).

        """
        #One way phase change
        phi=2*pi*((self.L+d)*f/c + d/self.wl)

        return self.t1/(1-self.r1*self.r2*np.exp(-2j*phi))

#}}}

#{{{ Often used parameters

def finesse(r1, r2, power=False):
    '''
    Returns the finesse of a cavity

    Parameters
    ----------
    r1 : float
        Reflectivity of the first mirror.
    r2 : float
        Reflectivity of the second mirror.
    power : boolean, optional
        If True, r1 and r2 are treated as power reflectivities.
        Otherwise, r1 and r2 are regarded as amplitude reflectivities.
        Defaults False.
    '''
    if power:
        r1 = np.sqrt(r1)
        r2 = np.sqrt(r2)

    return pi*np.sqrt(r1*r2)/(1-r1*r2)

#}}}
