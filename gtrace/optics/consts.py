#{{{ Imports

import numpy as np

#}}}

#{{{ Index of refractions

def sellmeier(wl, B1, B2, B3, C1, C2, C3):
    '''
    Calculate index of refraction using Sellmeiers equation

    n^2 = 1+B1*wl^2/(wl^2 - C1) + B2*wl^2/(wl^2 - C2) + B3*wl^2/(wl^2 - C3)

    See below for the coefficients for specific materials.
    http://www.cvimellesgriot.com/products/Documents/Catalog/Dispersion_Equations.pdf
    '''

    n = np.sqrt(1+B1*wl**2/(wl**2 - C1) + B2*wl**2/(wl**2 - C2) + B3*wl**2/(wl**2 - C3))
    return n

def n_fused_silica(wl):
    '''
    Calculate the index of refraction of fused silica for a given wavelength.
    '''
    #Convert the wavelength to the unit of um.
    wl = wl*1e6

    B1 = 6.961663e-1
    B2 = 4.079426e-1
    B3 = 8.974794e-1
    C1 = 4.67914826e-3
    C2 = 1.35120631e-2
    C3 = 9.79340025e1

    return sellmeier(wl, B1, B2, B3, C1, C2, C3)

def n_sapphire_ordinary(wl):
    '''
    Calculate the index of refraction of Sapphire ordinary axis for a given wavelength.
    '''

    #Convert the wavelength to the unit of um.
    wl = wl*1e6

    B1 = 1.4313493
    B2 = 6.5054713e-1
    B3 = 5.3414021
    C1 = 5.2799261e-3
    C2 = 1.42382647e-2
    C3 = 3.25017834e2

    return sellmeier(wl, B1, B2, B3, C1, C2, C3)

def n_sapphire_extraordinary(wl):
    '''
    Calculate the index of refraction of Sapphire extraordinary axis for a given wavelength.
    '''

    #Convert the wavelength to the unit of um.
    wl = wl*1e6

    B1 = 1.5039759
    B2 = 5.5069141e-1
    B3 = 6.5927379
    C1 = 5.48041129e-3
    C2 = 1.47994281e-2
    C3 = 4.0289514e2

    return sellmeier(wl, B1, B2, B3, C1, C2, C3)

n_fused_silica_532nm = 1.46071
n_fused_silica_1064nm = 1.45
n_fused_silica_1550nm = 1.444

no_sapphire_532nm = 1.77170
ne_sapphire_532nm = 1.76355

no_sapphire_1064nm = 1.75449
ne_sapphire_1064nm = 1.74663

no_sapphire_1550nm = 1.74618
ne_sapphire_1550nm = 1.73838


#}}}
