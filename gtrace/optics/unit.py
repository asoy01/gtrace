#{{{ Imports

import numpy as np
pi = np.pi

#}}}

#{{{ Length units
km = 1e3
m = 1.0
cm = 0.01
mm = 1e-3
um = 1e-6
nm = 1e-9
#}}}

#{{{ Power units

kW = 1e3
W = 1.0
mW = 1e-3
uW = 1e-6

#}}}

#{{{ Frequencies

THz = 1e12
GHz = 1e9
MHz = 1e6
kHz = 1e3
Hz = 1.0
mHz = 1e-3
uHz = 1e-6

#}}}

#{{{ Other

ppm = 1e-6

#}}}

#{{{ Conversion Functions

def rad2deg(rad):
    return rad * 180.0 / pi

def deg2rad(deg):
    return deg * pi /180.0

#}}}
