#{{{ Import
import numpy as np
pi = np.pi
#}}}

#{{{ Snell's Law

def deflection_angle(theta, n1, n2, deg=True):
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

    A line is specified by its origin (pos) and the
    direction vector (dirVect).
    A plane is specfied by its center coordinates (plane_center)
    and the normal vector (normalVector).
    The plane has its size (diameter).

    The returned value is a dictionary with the following keys:
    "Intersection Point": numpy array of the coordinates of the intersection point.
    "isHit": A boolean value of whether the line intersects with the plane or not.
    "distance": Distance between the origin of the line and the intersection point.
    "distance from center": Distance between the center of the plane and the intersection point.
    '''

    #Make sure the inputs are ndarrays
    pos = np.array(pos, dtype=np.float64)
    dirVect = np.array(dirVect, dtype=np.float64)
    plane_center = np.array(plane_center, dtype=np.float64)
    normalVector = np.array(normalVector, dtype=np.float64)
    diameter = float(diameter)
    
    #Get a normalized vector along the plane
    plVect = np.array([-normalVector[1], normalVector[0]])
    plVect = plVect/np.linalg.norm(plVect)

    #Normalize
    dirVect = dirVect/np.linalg.norm(dirVect)

    #Make sure that the plVect and dirVect are not parallel
    if np.abs(np.dot(dirVect, plVect)) > 1 - 1e-10:
        return {'Intersection Point': np.array((0.,0.)), 'isHit': False,
            'distance': 0.0,
            'distance from center': 0.0}

    
    #Solve line equations to get the intersection point
    M = np.vstack((dirVect, -plVect)).T
    ans = np.linalg.solve(M, plane_center - pos)    
    intersection_point = pos + ans[0]*dirVect

    #How far the intersection point is from the center
    #of the plane
    dist_from_center = np.abs(ans[1])
    if dist_from_center > diameter/2.0\
           or ans[0] < 0.\
           or np.dot(dirVect, normalVector) > 0.:
        
        hit = False
    else:
        hit = True
        
    return {'Intersection Point': intersection_point, 'isHit': hit,
            'distance': np.abs(ans[0]),
            'distance from center': ans[1]}


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
    
    pos: Origin of the line
    dirVect: Direction of the line
    chord_center: The center of the chord made by the arc.
    chordNormVect: Normal vector of the chord.
    invROC: Inverse of the ROC of the arc. Positive for concave surface.
    diameter: Length of the chord.
    '''
    #Make sure the inputs are ndarrays
    pos = np.array(pos, dtype=np.float64)
    dirVect = np.array(dirVect, dtype=np.float64)
    chord_center = np.array(chord_center, dtype=np.float64)
    chordNormVect = np.array(chordNormVect, dtype=np.float64)
    invROC = float(invROC)
    diameter = float(diameter)

    #Normalize
    dirVect = dirVect/np.linalg.norm(dirVect)
    chordNormVect = chordNormVect/np.linalg.norm(chordNormVect)

    #Check if the ROC is too large.
    if np.abs(invROC) < 1e-5:
        #It is almost a plane
        ans = line_plane_intersection(pos, dirVect, chord_center, chordNormVect, diameter)
        localNormVect = chordNormVect
        localNormAngle = np.mod(np.arctan2(localNormVect[1],
                                           localNormVect[0]), 2*pi)

        ans['localNormVect'] = localNormVect
        ans['localNormAngle'] = localNormAngle

        return ans

    ROC = 1/invROC
    
    
    #Compute the center of the arc
    theta = np.arcsin(diameter/(2*ROC))
    l = ROC*np.cos(theta)
    arc_center = chord_center + chordNormVect*l

    #For convex surface, pos has to be outside the circle.
    if ROC < 0 and np.linalg.norm(pos - arc_center) < np.abs(ROC):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}


    #First, decompose the vector connecting from the arc_center
    #to pos into the components parallel to the line and orthogonal to it.
    # s is the component in the orthogonal direction and t is the one along
    #the line.
    #A vector orthogonal to the line
    k = np.array([-dirVect[1], dirVect[0]])
    #Solve the equation to decompose the vector pos-arc_center
    M = np.vstack((k, -dirVect)).T
    ans = np.linalg.solve(M, pos - arc_center)
    s = ans[0]
    t = ans[1]
    
    if np.abs(s) > np.abs(ROC):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}

    #Compute two cross points
    #Work with the chord formed by the line and the circle.
    #d is half the length of the chord.
    d = np.sqrt(ROC**2 - s**2)
    if ROC > 0:
        intersection_point = k*s+arc_center + d*dirVect
        localNormVect = arc_center - intersection_point
    else:
        intersection_point = k*s+arc_center - d*dirVect
        localNormVect = intersection_point - arc_center

    #Check if dirVect and the vector connecting from pos to intersection_point
    #are pointing the same direction.
    if np.dot(dirVect, intersection_point - pos) < 0:
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}
        
    #Normalize
    localNormVect = localNormVect/np.linalg.norm(localNormVect)
    localNormAngle = np.mod(np.arctan2(localNormVect[1],
                                       localNormVect[0]), 2*pi)

    #Check if the intersection point is within the
    #diameter
    v0 = - np.sign(ROC) * chordNormVect*(1-1e-16)  #(1-1e-16) is necessary to avoid rounding error
    v1 = intersection_point - arc_center
    v1 = v1/np.linalg.norm(v1)*(1-1e-16)
    if np.arccos(np.dot(v0,v1)) > np.abs(theta):
        if verbose:
            print('The line does not hit the arc.')
        return {'isHit': False}

    distance = np.linalg.norm(intersection_point - pos)


    
    return {'Intersection Point': intersection_point, 'isHit': True,
            'distance': distance, 'localNormVect': localNormVect,
            'localNormAngle': localNormAngle}

#}}}

#{{{ vector_rotation_2D

def vector_rotation_2D(vect, angle):
    vect = np.array(vect)
    angle = float(angle)
    
    M = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),np.cos(angle)]])
    return np.dot(M, vect)

#}}}

def vector_normalize(vect):
    '''
    Normalize a vector
    '''
    
    return vect/np.linalg.norm(vect)

#{{{ normSpheric

def normSpheric(normAngle, invROC, dist_from_center):
    '''
    Returns the local normal angle of a spheric mirror
    at a distance from the center.

    normAngle: The angle formed by the normal vector of the mirror
               at the center and the x-axis.

    invROC: 1/R, where R is the ROC of the mirror.

    dist_from_center: The distance from the center of the point where
                      the local normal is requested.
                      This is a signed value.
                      For a mirror facing +x (the normal vector points
                      towards positive x direction), this distance
                      is positive for points with positive y coordinate,
                      and negative for points with negative y coordinate.
    '''

    normAngle = np.mod(normAngle, 2*pi)
    return np.mod(np.arcsin(- dist_from_center * invROC) + normAngle, 2*pi)

#}}}

#{{{ reflection and deflection angle

def refl_defl_angle(beamAngle, normAngle, n1, n2, invROC=None):
    '''
    Returns a tuples of reflection and deflection angles.
    
    beamAngle: The angle formed by the propagation direction vector
               of the incident beam and the x-axis.

    normAngle: The angle formed by the normal vector of the surface
               and the x-axis.

    n1: Index of refraction of the incident side medium.

    n2: Index of refraction of the transmission side medium.
    '''

    beamAngle = np.mod(beamAngle, 2*pi)
    normAngle = np.mod(normAngle, 2*pi)
    
    incidentAngle = np.mod(beamAngle - normAngle, 2*pi) - pi

    reflAngle = np.mod(normAngle - incidentAngle, 2*pi)
    
    deflAngle = np.arcsin(n1*np.sin(incidentAngle)/n2)
    deflAngle = np.mod(deflAngle + pi + normAngle, 2*pi)

    if not invROC == None:
        #Calculate ABCD matrices

        #Absolute value of the incident angle
        theta1 = np.abs(incidentAngle)

        #For reflection
        Mrx = np.array([[1., 0.], [-2*n1*invROC/np.cos(theta1), 1.]])
        Mry = np.array([[1., 0.], [-2*n1*invROC*np.cos(theta1), 1.]])

        #For transmission
        theta2 = np.arcsin(n1*np.sin(theta1)/n2)

        nex = (n2*np.cos(theta2)-n1*np.cos(theta1))/(np.cos(theta1)*np.cos(theta2))
        Mtx = np.array([[np.cos(theta2)/np.cos(theta1), 0.],
                        [nex*invROC, np.cos(theta1)/np.cos(theta2)]])

        ney = n2*np.cos(theta2)-n1*np.cos(theta1)
        Mty = np.array([[1., 0.],[ney*invROC, 1.]])

        return (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty)

    else:
        return (reflAngle, deflAngle)
    
    
#}}}

#{{{ reflection and deflection angle for cylindrical surface

def cyl_refl_defl_angle(beamAngle, normAngle, n1, n2, invROC=None, curve_direction='h'):
    '''
    Returns a tuples of reflection and deflection angles for incidence of a beam into a cylindrical surface.
    
    beamAngle: The angle formed by the propagation direction vector
               of the incident beam and the x-axis.

    normAngle: The angle formed by the normal vector of the surface
               and the x-axis.

    n1: Index of refraction of the incident side medium.

    n2: Index of refraction of the transmission side medium.

    invROC: Inverse of the radius of curvature.

    curve_direction: Direction of curvature. Either 'h' or 'v'.
    '''

    beamAngle = np.mod(beamAngle, 2*pi)
    normAngle = np.mod(normAngle, 2*pi)
    
    incidentAngle = np.mod(beamAngle - normAngle, 2*pi) - pi

    reflAngle = np.mod(normAngle - incidentAngle, 2*pi)
    
    deflAngle = np.arcsin(n1*np.sin(incidentAngle)/n2)
    deflAngle = np.mod(deflAngle + pi + normAngle, 2*pi)

    if not invROC == None:
        #Calculate ABCD matrices

        #Absolute value of the incident angle
        theta1 = np.abs(incidentAngle)

        #For reflection
        if curve_direction == 'h':
            Mrx = np.array([[1., 0.], [-2*n1*invROC/np.cos(theta1), 1.]])
            Mry = np.array([[1., 0.], [0., 1.]])
        else:
            Mrx = np.array([[1., 0.], [0., 1.]])
            Mry = np.array([[1., 0.], [-2*n1*invROC*np.cos(theta1), 1.]])

        #For transmission
        theta2 = np.arcsin(n1*np.sin(theta1)/n2)

        nex = (n2*np.cos(theta2)-n1*np.cos(theta1))/(np.cos(theta1)*np.cos(theta2))
        Mtx = np.array([[np.cos(theta2)/np.cos(theta1), 0.],
                        [nex*invROC, np.cos(theta1)/np.cos(theta2)]])

        ney = n2*np.cos(theta2)-n1*np.cos(theta1)
        Mty = np.array([[1., 0.],[ney*invROC, 1.]])

        return (reflAngle, deflAngle, Mrx, Mry, Mtx, Mty)

    else:
        return (reflAngle, deflAngle)
    
    
#}}}

#}}}

#{{{ VariCAD utility functions

def vc_deflect(theta, theta1, n1, n2):
    '''
    Deflection angle helper function for VariCAD.
    theta is the angle of the surface measured from right.
    theta1 is the angle of the incident beam measured from right.
    It returns an angle of the deflected beam measured from right.
    
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
    #Combert theta and theta1 to 0-360 format
    if theta < 0:
        theta = 360.0 + theta

    if theta > 180:
        theta = theta -180.0
        
    if theta1 < 0:
        theta1 = 360.0 + theta1

    return theta - (theta1 - theta)

#}}}
