#{{{ Import
import math
import numpy as np
pi = np.pi
#}}}

#{{{ Snell's Law

def deflection_angle(theta, n1, n2, deg=True):
    """Calculate deflection angle according to Snell's law.

    Parameters
    ----------
    theta : float
        Angle of incidence.
    n1 : float
        Refractive index of the first medium.
    n2 : float
        Refraction index of the second medium.
    deg : boolean, optional
        True if theta is specified in degrees.
    """
    if deg:
        factor = pi/180.0
    else:
        factor = 1.0

    return np.arcsin(n1*np.sin(theta*factor)/n2)/factor

#}}}

#{{{ Geometry utilities

#{{{ line_plane_intersection

def line_plane_intersection(pos,
                            dirVect,
                            plane_center,
                            normalVector,
                            diameter):
    '''
    Compute the intersection point between a line
    and a plane

    Parameters
    ----------
    pos : numpy.ndarray
        The position of the end point of the line.
    dirVert : numpy.ndarray
        The directional vector specifying the line.
    plane_center : numpy.ndarray
        The position of the center of the plane.
    normalVector: numpy.ndarray
        The normal vector of the plane.
    diameter: float
        The diameter of the plane.

    Returns
    -------
    dict
        The returned value is a dictionary with the following keys:
        "Intersection Point": numpy array of the coordinates of the intersection point.
        "isHit": A boolean value of whether the line intersects with the plane or not.
        "distance": Distance between the origin of the line and the intersection point.
        "distance from center": Distance between the center of the plane and the intersection point.

    Notes
    -----
    The vectors are read element by element and the arithmetic is done
    on plain floats. Every hit test comes through here, and at this
    size numpy costs more in call overhead than the twenty
    floating point operations are worth: a two-element ``norm`` alone
    takes a microsecond, and a 2x2 ``solve`` calls into LAPACK. The
    inputs may still be a list, a tuple or an array of any numeric
    type - nothing here does more than take ``[0]`` and ``[1]``.
    '''

    px = pos[0]
    py = pos[1]
    dx = dirVect[0]
    dy = dirVect[1]
    cx = plane_center[0]
    cy = plane_center[1]
    nx = normalVector[0]
    ny = normalVector[1]
    diameter = float(diameter)

    miss = {'Intersection Point': np.array((0.,0.)), 'isHit': False,
            'distance': 0.0,
            'distance from center': 0.0}

    #Get a normalized vector along the plane
    nlen = math.hypot(nx, ny)
    if nlen == 0.0:
        #A plane with no normal is not a plane.
        return miss
    lx = -ny/nlen
    ly = nx/nlen

    #Normalize
    dlen = math.hypot(dx, dy)
    if dlen == 0.0:
        #A line that goes nowhere reaches nothing.
        return miss
    dx = dx/dlen
    dy = dy/dlen

    #Make sure that the plVect and dirVect are not parallel
    if abs(dx*lx + dy*ly) > 1 - 1e-10:
        return miss

    #Solve the line equations to get the intersection point. Writing
    #pos + a*dirVect = plane_center + b*plVect as a 2x2 system and
    #applying Cramer's rule, which is what a solver does to a matrix
    #this small anyway.
    det = -dx*ly + lx*dy
    bx = cx - px
    by = cy - py
    a = (-bx*ly + lx*by)/det
    b = (dx*by - dy*bx)/det

    intersection_point = np.array((px + a*dx, py + a*dy))

    #How far the intersection point is from the center
    #of the plane
    dist_from_center = abs(b)
    if dist_from_center > diameter/2.0\
           or a < 0.\
           or dx*nx + dy*ny > 0.:

        hit = False
    else:
        hit = True

    return {'Intersection Point': intersection_point, 'isHit': hit,
            'distance': abs(a),
            'distance from center': b}


#}}}

#{{{ line_arc_intersection

def line_arc_intersection(pos,
                          dirVect,
                          chord_center,
                          chordNormVect,
                          invROC,
                          diameter,
                          verbose=False):
    '''
    Compute the intersection point between a line
    and an arc.

    Parameters
    ----------
    pos : numpy.ndarray
        Origin of the line.
    dirVect : numpy.ndarray
        Direction of the line.
    chord_center : numpy.ndarray
        The center of the chord made by the arc.
    chordNormVect : numpy.ndarray
        Normal vector of the chord.
    invROC : float
        Inverse of the ROC of the arc. Positive for concave surface.
    diameter : float
        Length of the chord.
    verbose : boolean, optional
        Prints useful information.

    Returns
    -------
    dict
        The returned value is a dictionary with the following keys:
        "Intersection Point": numpy array of the coordinates of the intersection point.
        "isHit": A boolean value of whether the line intersects with the plane or not.
        "distance": Distance between the origin of the line and the intersection point.
        "localNormVect": localNormVect,
        "localNormAngle": localNormAngle.

    Notes
    -----
    Read element by element and worked out on plain floats, for the
    reason given in line_plane_intersection.
    '''
    px = pos[0]
    py = pos[1]
    dx = dirVect[0]
    dy = dirVect[1]
    ccx = chord_center[0]
    ccy = chord_center[1]
    cnx = chordNormVect[0]
    cny = chordNormVect[1]
    invROC = float(invROC)
    diameter = float(diameter)

    #Normalize
    dlen = math.hypot(dx, dy)
    if dlen == 0.0:
        #A line that goes nowhere reaches nothing.
        return {'isHit': False}
    dx = dx/dlen
    dy = dy/dlen
    cnlen = math.hypot(cnx, cny)
    if cnlen == 0.0:
        return {'isHit': False}
    cnx = cnx/cnlen
    cny = cny/cnlen

    #Check if the ROC is too large.
    if abs(invROC) < 1e-5:
        #It is almost a plane
        ans = line_plane_intersection((px, py), (dx, dy), (ccx, ccy),
                                      (cnx, cny), diameter)
        ans['localNormVect'] = np.array((cnx, cny))
        ans['localNormAngle'] = math.atan2(cny, cnx) % (2*pi)

        return ans

    ROC = 1/invROC


    #Compute the center of the arc
    sin_theta = diameter/(2*ROC)
    if -1.0 <= sin_theta <= 1.0:
        theta = math.asin(sin_theta)
    else:
        #A face of more than a hemisphere. numpy answered that with a
        #nan and the tests below then let the point through; keep it
        #that way rather than raising where nothing used to raise.
        theta = float('nan')
    l = ROC*math.cos(theta)
    acx = ccx + cnx*l
    acy = ccy + cny*l

    #For convex surface, pos has to be outside the circle.
    if ROC < 0 and math.hypot(px - acx, py - acy) < abs(ROC):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}


    #First, decompose the vector connecting from the arc_center
    #to pos into the components parallel to the line and orthogonal to it.
    # s is the component in the orthogonal direction and t is the one along
    #the line.
    #A vector orthogonal to the line
    kx = -dy
    ky = dx
    #Decompose the vector pos-arc_center. It is a 2x2 system whose
    #determinant is the squared length of the direction vector, and
    #only the orthogonal component is used.
    bx = px - acx
    by = py - acy
    s = (dx*by - dy*bx)/(dx*dx + dy*dy)

    if abs(s) > abs(ROC):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}

    #Compute two cross points
    #Work with the chord formed by the line and the circle.
    #d is half the length of the chord.
    d = math.sqrt(ROC*ROC - s*s)
    if ROC > 0:
        ipx = kx*s + acx + d*dx
        ipy = ky*s + acy + d*dy
        lnx = acx - ipx
        lny = acy - ipy
    else:
        ipx = kx*s + acx - d*dx
        ipy = ky*s + acy - d*dy
        lnx = ipx - acx
        lny = ipy - acy

    #Check if dirVect and the vector connecting from pos to intersection_point
    #are pointing the same direction.
    if dx*(ipx - px) + dy*(ipy - py) < 0:
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}

    #Normalize
    lnlen = math.hypot(lnx, lny)
    lnx = lnx/lnlen
    lny = lny/lnlen
    localNormAngle = math.atan2(lny, lnx) % (2*pi)

    #Check if the intersection point is within the
    #diameter
    sgn = 1.0 if ROC > 0 else -1.0
    v0x = -sgn*cnx*(1-1e-16)   #(1-1e-16) is necessary to avoid rounding error
    v0y = -sgn*cny*(1-1e-16)
    v1len = math.hypot(ipx - acx, ipy - acy)
    v1x = (ipx - acx)/v1len*(1-1e-16)
    v1y = (ipy - acy)/v1len*(1-1e-16)
    cosine = v0x*v1x + v0y*v1y
    #Those factors keep this inside the domain of acos, but only just;
    #hold it there so that rounding cannot raise where numpy used to
    #return a nan.
    if cosine > 1.0:
        cosine = 1.0
    elif cosine < -1.0:
        cosine = -1.0
    if math.acos(cosine) > abs(theta):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}

    distance = math.hypot(ipx - px, ipy - py)



    return {'Intersection Point': np.array((ipx, ipy)), 'isHit': True,
            'distance': distance, 'localNormVect': np.array((lnx, lny)),
            'localNormAngle': localNormAngle}

#}}}

#{{{ vector_rotation_2D

def vector_rotation_2D(vect, angle):
    """Rotate a 2D vector by an angle.

    Parameters
    ----------
    vect : numpy.ndarray
        A 2D vector.
    angle : float
        Angle of rotation in radians.

    Returns
    -------
    numpy.ndarray
        The rotated vector.

    Notes
    -----
    A single 2-vector is turned with four multiplications, which is far
    cheaper than building a matrix to multiply it by. Anything else -
    an array of shape (2, N), which is how the drawing code hands over
    a whole outline at once - goes through the matrix as before.
    """
    ca = math.cos(angle)
    sa = math.sin(angle)

    try:
        plain = len(vect) == 2 and not hasattr(vect[0], '__len__')
    except TypeError:
        plain = False
    if plain:
        x = vect[0]
        y = vect[1]
        return np.array((ca*x - sa*y, sa*x + ca*y))

    M = np.array([[ca, -sa],
                  [sa, ca]])
    return np.dot(M, np.array(vect))

#}}}

def vector_normalize(vect):
    '''
    Normalize a vector

    Parameters
    ----------
    vect : numpy.ndarray
        The vector to be normalized

    Returns
    -------
    numpy.ndarray
        The normalized vector.
    '''

    return vect/np.linalg.norm(vect)

#{{{ normSpheric

def normSpheric(normAngle, invROC, dist_from_center):
    '''
    Returns the local normal angle of a spheric mirror
    at a distance from the center.

    Parameters
    ----------
    normAngle : float
        The angle formed by the normal vector of the mirror
        at the center and the x-axis.
    invROC : float
        1/R, where R is the ROC of the mirror.
    dist_from_center: float
        The distance from the center of the point where
        the local normal is requested.
        This is a signed value.
        For a mirror facing +x (the normal vector points
        towards positive x direction), this distance
        is positive for points with positive y coordinate,
        and negative for points with negative y coordinate.

    Returns
    -------
    float
        The local normal angle of a spheric mirror
        at a distance from the center.
    '''

    normAngle = np.mod(normAngle, 2*pi)
    return np.mod(np.arcsin(- dist_from_center * invROC) + normAngle, 2*pi)

#}}}

#{{{ reflection and deflection angle

def _surface_angles(beamAngle, normAngle):
    '''
    The reflection and deflection geometry of a surface: the incident
    angle, and the angle of the reflected ray.

    Split out so that the spherical and cylindrical entry points below
    cannot drift apart. They differ only in which plane gets the
    curvature, and everything else - these angles and the matrices in
    _surface_matrices - is common to both.
    '''
    two_pi = 2*pi
    beamAngle = beamAngle % two_pi
    normAngle = normAngle % two_pi
    incidentAngle = (beamAngle - normAngle) % two_pi - pi
    reflAngle = (normAngle - incidentAngle) % two_pi
    return incidentAngle, reflAngle

def _surface_matrices(theta1, n1, n2, invROC_x, invROC_y):
    '''
    The ABCD matrices of a surface at incidence theta1, given the
    curvature each transverse plane sees.

    Two curvatures rather than one, because that is the only difference
    between a spherical surface and a cylindrical one: a sphere presents
    the same curvature to both planes, a cylinder presents it to one and
    nothing to the other. Writing it once this way is deliberate - the
    cylindrical transmission matrices were previously a copy of the
    spherical ones and had never been given the distinction at all.

    x is the plane of incidence (tangential), y is perpendicular to it
    (sagittal). The forms are Siegman, Lasers, Table 15.1 (d) for
    reflection and (f), (g) for refraction, in the reduced-slope
    convention where every determinant is 1.

    Parameters
    ----------
    theta1 : float
        Angle of incidence, from the normal, unsigned.
    n1, n2 : float
        Index on the incident side and on the far side.
    invROC_x, invROC_y : float
        The inverse radius of curvature each plane sees. Zero leaves
        that plane without power - which for reflection is the identity,
        and for refraction is still not: a tilted flat interface changes
        the width of the beam in the plane of incidence, and any
        interface carries the index change.

    Returns
    -------
    (Mrx, Mry, Mtx, Mty)

    Notes
    -----
    Past the critical angle there is no transmitted ray, and the two
    transmission matrices come back full of nan. That is how a caller
    tells total internal reflection from an ordinary refraction, so it
    is produced deliberately here: math.asin raises on an argument
    outside [-1, 1], where the numpy call this replaces returned a nan.
    '''
    cos1 = math.cos(theta1)

    #For reflection. R_e = R*cos(theta) in the plane of incidence and
    #R/cos(theta) perpendicular to it, so the same curvature focuses
    #more strongly in the plane of incidence.
    Mrx = np.array([[1., 0.], [-2*n1*invROC_x/cos1, 1.]])
    Mry = np.array([[1., 0.], [-2*n1*invROC_y*cos1, 1.]])

    #For transmission
    sin2 = n1*math.sin(theta1)/n2
    if -1.0 <= sin2 <= 1.0:
        theta2 = math.asin(sin2)
        cos2 = math.cos(theta2)
    else:
        #Total internal reflection: nothing is transmitted.
        cos2 = float('nan')

    nex = (n2*cos2-n1*cos1)/(cos1*cos2)
    Mtx = np.array([[cos2/cos1, 0.],
                    [nex*invROC_x, cos1/cos2]])

    ney = n2*cos2-n1*cos1
    Mty = np.array([[1., 0.], [ney*invROC_y, 1.]])

    return Mrx, Mry, Mtx, Mty

def refl_defl_angle(beamAngle, normAngle, n1, n2, invROC=None):
    '''
    Returns a tuples of reflection and deflection angles.

    Parameters
    ----------
    beamAngle : float
        The angle formed by the propagation direction vector
        of the incident beam and the x-axis.
    normAngle : float
        The angle formed by the normal vector of the surface
        and the x-axis.
    n1 : float
        Index of refraction of the incident side medium.
    n2 : float
        Index of refraction of the transmission side medium.
    invROC : float or None, optional
        Inverse of the radius of curvature.

    Returns
    -------
    6-tuple or 2-tuple
    (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) or (reflAngle, deflAngle)
    '''

    incidentAngle, reflAngle = _surface_angles(beamAngle, normAngle)

    deflAngle = np.arcsin(n1*np.sin(incidentAngle)/n2)
    deflAngle = np.mod(deflAngle + pi + np.mod(normAngle, 2*pi), 2*pi)

    if not invROC == None:
        #A sphere presents the same curvature to both planes.
        theta1 = np.abs(incidentAngle)
        Mrx, Mry, Mtx, Mty = _surface_matrices(theta1, n1, n2, invROC, invROC)
        return (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty)

    else:
        return (reflAngle, deflAngle)


#}}}

#{{{ reflection and deflection angle for cylindrical surface

def cyl_refl_defl_angle(beamAngle, normAngle, n1, n2, invROC=None, curve_direction='h'):
    '''
    Returns a tuples of reflection and deflection angles for incidence of a beam into a cylindrical surface.

    Parameters
    ----------
    beamAngle : float
        The angle formed by the propagation direction vector
        of the incident beam and the x-axis.
    normAngle : float
        The angle formed by the normal vector of the surface
        and the x-axis.
    n1 : float
        Index of refraction of the incident side medium.
    n2 : float
        Index of refraction of the transmission side medium.
    invROC : float or None, optional
        Inverse of the radius of curvature of the curved plane. The
        other plane is flat, whatever this says.
    curve_direction : str, optional
        Which plane carries the curvature. 'h' is the plane of the
        trace, 'v' is perpendicular to it.

    Returns
    -------
    6-tuple or 2-tuple
    (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty) or (reflAngle, deflAngle)

    Notes
    -----
    A cylinder gives its curvature to one plane and nothing to the
    other, so the matrices are those of a surface that is curved in the
    one and flat in the other. This is *not* the same as a zero matrix
    in the flat plane, which is why it is expressed as a curvature of
    zero rather than as a special case: reflection off a flat surface
    is the identity, but refraction through one is not - a tilted flat
    interface still changes the width of the beam in the plane of
    incidence, and any interface carries the index change.
    '''

    incidentAngle, reflAngle = _surface_angles(beamAngle, normAngle)

    deflAngle = np.arcsin(n1*np.sin(incidentAngle)/n2)
    deflAngle = np.mod(deflAngle + pi + np.mod(normAngle, 2*pi), 2*pi)

    if not invROC == None:
        theta1 = np.abs(incidentAngle)
        invROC_x = invROC if curve_direction == 'h' else 0.0
        invROC_y = invROC if curve_direction == 'v' else 0.0
        Mrx, Mry, Mtx, Mty = _surface_matrices(theta1, n1, n2,
                                               invROC_x, invROC_y)
        return (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty)

    else:
        return (reflAngle, deflAngle)


#}}}

#}}}

#{{{ VariCAD utility functions

def vc_deflect(theta, theta1, n1, n2):
    '''
    Deflection angle helper function for VariCAD.

    Parameters
    ----------
    theta : float
        Angle of the surface measured from right.
    theta1 : float
        Angle of the incident beam measured from right.
    n1 : float
        Index of refraction of the incident side medium.
    n2 : float
        Index of refraction of the transmission side medium.

    Returns
    -------
    phi2 : float
        Angle of the deflected beam measured from right.

    '''
    #Combert theta and theta1 to 0-360 format
    if theta < 0:
        theta = 360.0 + theta

    if theta > 180:
        theta = theta -180.0

    if theta1 < 0:
        theta1 = 360.0 + theta1

    #Determine the incident angle
    phi = abs(theta - theta1)
    phi1 = 90.0-np.arcsin(np.abs(np.sin(pi*phi/180.0)))*180.0/pi

    #Calculate deflection angle
    phi2 = deflection_angle(phi1, n1, n2)

    #Convert to the 0-360 angle
    s1 = np.sign(np.sin(pi*(theta1 - theta)/180.0))
    s2 = -np.sign(np.cos(pi*(theta1 - theta)/180.0))
    phi2 = theta + s1*90 + s1*s2*phi2
    return phi2


def vc_reflect(theta, theta1):
    """Convert theta and theta1 to 0-360 format.

    Parameters
    ----------
    theta : float
        Angle of the surface measured from right.
    theta1 : float
        Angle of the incident beam measured from right.

    Returns
    -------
    float
    """
    #Combert theta and theta1 to 0-360 format
    if theta < 0:
        theta = 360.0 + theta

    if theta > 180:
        theta = theta -180.0

    if theta1 < 0:
        theta1 = 360.0 + theta1

    return theta - (theta1 - theta)

#}}}
