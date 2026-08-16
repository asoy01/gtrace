Basic Concepts
===============================

This page describes the objects a layout is made of: the plane they sit on,
the beams, and the elements the beams hit. :doc:`intro` shows how they work
together. This page says what each one is.

The plane
---------------

An optical system lies on a two dimensional plane, and a place on it is a
pair of Cartesian coordinates (x, y). The x axis runs to the right and the
y axis upward. You decide where to put the origin. Coordinates on either
side of it are equally good. A layout can run from x = -3 m to x = +3 m, if
that is where its elements are.

Lengths are in metres and angles are in radians. ``gtrace.unit`` holds
multipliers so that you never write either in full::

    import gtrace.optcomp as opt
    from gtrace.beam import GaussianBeam
    from gtrace.layout import q_from_waist
    from gtrace.unit import *        # mm, cm, inch, nm, um, ppm, deg2rad, ...

    M = opt.Mirror(HRcenter=[30*cm, 0.0], diameter=2*inch,
                   thickness=10*mm, normAngleHR=deg2rad(170))

The examples on the rest of this page use these imports.

Direction
-----------

.. image:: imgs/Direction.*
   :height: 5cm

A direction is the way a mirror faces, or the way a beam travels. You can
give it in two ways: as an angle measured from the x axis counterclockwise,
or as a 2D vector. Every object that has a direction has both. Setting one
updates the other::

    >>> b = GaussianBeam(wl=1064*nm, dirAngle=deg2rad(30))
    >>> b.dirVect
    array([0.8660254, 0.5      ])

    >>> b.dirVect = [3.0, 0.0]        # any length will do
    >>> b.dirVect                     # it is normalised for you
    array([1., 0.])
    >>> b.dirAngle
    0.0

Write to the angle and the vector changes with it. Write to the vector and
the angle changes with it. A
:py:class:`GaussianBeam<gtrace.beam.GaussianBeam>` holds this pair as
``dirAngle`` and ``dirVect``. A :py:class:`Mirror<gtrace.optcomp.Mirror>`
holds it as ``normAngleHR`` and ``normVectHR``, for the normal of its front
face.

Beam
-----------

.. image:: imgs/Beam.*
   :height: 10cm

A Gaussian beam is an instance of
:py:class:`GaussianBeam<gtrace.beam.GaussianBeam>`. It starts at ``pos``,
travels along ``dirVect``, and is drawn for ``length`` metres::

    b = GaussianBeam(q0=q_from_waist(0.4*mm, 0.0, 1064*nm), wl=1064*nm,
                     pos=[0.0, 0.0], dirAngle=0.0, P=1.0, name='b')

Its shape is the complex beam parameter q at the point it starts from. The
x and y directions of the cross section are carried separately, as ``qx``
and ``qy``. They do not stay equal. A tilted or wedged surface acts
differently in the plane of incidence and out of it, so a beam that starts
round does not stay round. x is the direction in the plane of the drawing,
and y is out of it. ``q`` on its own reports the best matching circular
mode.

The rest of what a beam carries follows from those:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Meaning
   * - ``wx``, ``wy``
     - The beam radius at ``pos``, computed from ``qx`` and ``qy``.
   * - ``wl``, ``n``
     - Wavelength in vacuum, and the refractive index of the medium the
       beam is travelling through.
   * - ``P``
     - Power.
   * - ``optDist``
     - Optical distance accumulated since the source, glass included.
   * - ``Gouyx``, ``Gouyy``
     - Accumulated Gouy phase.
   * - ``Mx``, ``My``
     - The product of every ABCD matrix applied to the beam so far.
   * - ``name``, ``layer``
     - What the beam is called, and which DXF layer it is drawn on.

:py:meth:`waist()<gtrace.beam.GaussianBeam.waist>` reports where the waist
of the beam is and how big it is, measured from ``pos``. A negative
position means the beam has already passed the waist.
:doc:`propagation` covers how to move a beam, and how to hit an element
with it.

Mirror
-----------

.. image:: imgs/Mirror.png
   :height: 10cm

:py:class:`Mirror<gtrace.optcomp.Mirror>` is the basic optical element. It
is a piece of substrate with two surfaces, so it also serves for a
transparent window, a prism, a lens, or a light absorbing plate such as
black glass. Set the reflectivity and the transmission of each surface to
say which of those you mean.

The front surface is called HR and the back one AR, after the coatings they
usually carry. Each has its own position, curvature and reflectivity:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Meaning
   * - ``HRcenter``, ``ARcenter``
     - The apex of each face, where it crosses the optical axis. These are
       the points you place an element by.
   * - ``normAngleHR``, ``normVectHR``
     - Which way the front face points. The back face follows it, turned by
       ``wedgeAngle``.
   * - ``inv_ROC_HR``, ``inv_ROC_AR``
     - Curvature, held as the *inverse* radius, so a flat surface is zero
       and needs no special case. The viewer shows you the radius itself
       and converts on the way in and out.
   * - ``diameter``, ``thickness``
     - The aperture, and the thickness of the substrate measured at the rim.
   * - ``Refl_HR``, ``Trans_HR``, ``Refl_AR``, ``Trans_AR``
     - The fraction of the power each face reflects and transmits. A beam's
       power is multiplied by these as it goes. They are set independently,
       and gtrace does not require a pair of them to add up to 1.
   * - ``n``
     - Refractive index of the substrate.

Curved surfaces are spherical. For a cylindrical one, use
:py:class:`CyMirror<gtrace.optcomp.CyMirror>`, described under
:ref:`cylindrical-surfaces`. For a lens,
:py:class:`Lens<gtrace.optcomp.Lens>` is the same substrate ordered by
focal length instead of by the radii of its two faces.

Lens
-----------

A lens is a substrate whose two faces both refract, which a
:py:class:`Mirror<gtrace.optcomp.Mirror>` with two curved surfaces and a
low reflectivity already is. What :py:class:`Lens<gtrace.optcomp.Lens>`
adds is the way you order one: by focal length::

    from gtrace.optcomp import Lens
    from gtrace.unit import mm, inch

    L = Lens(f=500*mm)                                    # biconvex, 1 inch
    L = Lens(f=-100*mm, thickness=3*mm)                   # f < 0 comes out biconcave
    L = Lens(f=150*mm, shape='convex-plano')              # curved front, flat back
    L = Lens(f=150*mm, shape='meniscus', ROC_HR=-60*mm)

``shape`` is spelt the way a catalogue spells it: the front face first, the
back face second. ``'plano-convex'`` and ``'convex-plano'`` are therefore
the same lens, turned around. If you leave ``shape`` out, you get an
equiconvex lens for a positive focal length and an equiconcave one for a
negative focal length. Asking for a shape that contradicts the sign of the
focal length raises an error.

The radii are solved as a *thick* lens, which is what the trace then sees:
the beam refracts at both faces, with the substrate in between. Radii taken
from the thin lens formula would be a few parts in a thousand off, and much
further off for a short focal length. ``thickness`` is the
:py:class:`Mirror<gtrace.optcomp.Mirror>` thickness, measured between the
two chord planes, so it is the thickness at the rim.
``center_thickness`` reports the distance between the apexes that a
catalogue would quote.

gtrace refuses a lens that cannot be made out of the blank it was given,
and the message carries the number you need. Three cases are refused: two
concave faces that would meet in the middle, a face steeper than its own
aperture, and a focal length that no substrate of that thickness can reach.

Both faces reflect nothing by default, so a lens makes no ghost beams. A
real lens does reflect. But several lenses together produce so many faint
ghosts that the picture becomes unreadable and the trace becomes slow. Most
of the time a lens is in a layout to bend the main beam. Ask for the ghosts
when you want them::

    L = Lens(f=500*mm, Refl_HR=0.005, Trans_HR=0.995,
             Refl_AR=0.005, Trans_AR=0.995)

Those ghosts are counted as ghosts. A reflection off either face of a lens
raises the stray order of the beam. A reflection off the HR face of a
mirror does not. See :ref:`stray-order`.

The focal length is not stored. Reading ``f`` computes it from the
curvatures, the thickness and the index as they stand. Assigning to it
reshapes the faces to match, keeping the shape of the lens and leaving it
where it is. Tuning a lens against a mode matching target is therefore a
loop::

    for f in np.arange(150, 400, 10)*mm:
        L.f = f
        layout.trace()
        ...

.. _changing-a-curvature:

Moving and reshaping an element
--------------------------------

A curved surface is an arc. An arc has two natural reference points: the
apex, where it crosses the optical axis, and the centre of the chord it
spans, out at the rim. They lie a *sagitta* apart, and the sagitta depends
on the radius. ``HRcenter`` is the apex and ``HRcenterC`` the chord centre;
``ARcenter`` and ``ARcenterC`` are the same pair on the back face. The
thickness of the substrate is measured between the two chord planes, so it
is the thickness at the rim.

Assigning a new curvature therefore has to move one of the two points.
Which one it should be depends on what you are doing::

    M.inv_ROC_HR = 1.0/newROC

By default the *apex stays put* and the substrate slides back behind it.
This suits a reflective telescope, which is tuned by sweeping the radii of
its mirrors to get the magnification right. The layout puts the beam spot
on the HR surface, and the sweep is meant to change the beam size, not to
move the beam::

    for R in np.arange(-30, -20, 0.5):
        MMT1.inv_ROC_HR = 1.0/R
        layout.trace()          # the beam still lands where it did

For an optics the beam passes *through*, that is the wrong choice. There is
no spot on a surface to keep still, and what is bolted to the bench is the
substrate. Setting ``anchor_point`` to ``'center'`` keeps the middle of the
substrate where it is and lets the faces move on it::

    M.anchor_point = 'center'
    M.inv_ROC_HR = 1.0/newROC   # M.center unchanged, M.HRcenter moves

:py:class:`Lens<gtrace.optcomp.Lens>` defaults to ``'center'`` for this
reason; :py:class:`Mirror<gtrace.optcomp.Mirror>` and
:py:class:`CyMirror<gtrace.optcomp.CyMirror>` default to ``'HRcenter'``.

Either way the rest of the substrate follows. The far face, the sides and
the centre are all recomputed, so the body stays closed and the tracer and
the drawing agree about where it is. The anchor only chooses which end of
the sagitta is held fixed while that happens.

The anchor point is also what the element *turns about*. Assigning
``normAngleHR`` or ``normVectHR`` rotates the substrate rigidly about the
anchor, and so does ``rotate()`` by default. A mirror pivots the apex of
its HR face, so steering it does not walk the beam spot off it, which is
what ``rotate()`` has always done. A lens pivots the middle of its
substrate, so turning it does not carry it up the bench.
``rotate(a, center=True)`` pivots the middle whatever the anchor says, and
``rotate(a, center=[x, y])`` a given point.

Changing ``diameter`` goes through the same machinery, since the sagitta
depends on the aperture as well as on the radius. Changing ``inv_ROC_AR``
never moves the substrate at all: the back face has no spot to keep still,
so its chord plane stays and only its apex moves.

.. _cylindrical-surfaces:

Cylindrical surfaces
---------------------

:py:class:`CyMirror<gtrace.optcomp.CyMirror>` is the same substrate with
cylindrical faces instead of spherical ones. ``curve_direction`` says which
plane the cylinder curves in: ``'h'`` in the plane of the trace, ``'v'``
out of it. The other plane is flat, and a beam passing through it comes out
with the divergence it went in with.

Two things follow from working in 2D. First, only ``'h'`` is visible in the
drawing. A ``'v'`` mirror is drawn with straight faces. The plane of the
trace cuts a straight line out of it, and its focusing happens out of the
page.

Second, the two directions are not a relabelling of each other. At an angle
of incidence :math:`\theta`, a curved surface presents an effective radius
:math:`R\cos\theta` in the plane of incidence, and :math:`R/\cos\theta`
perpendicular to it. An ``'h'`` mirror of radius :math:`R` therefore has a
focal length of :math:`R\cos\theta/2`, and a ``'v'`` mirror of the same
radius has :math:`R/(2\cos\theta)`. At 45 degrees they differ by a factor
of two. Only at normal incidence do the two agree.

Transmission through a cylindrical face is worth stating separately,
because it is easy to assume more than is true. The uncurved plane loses
the *power* of the surface and nothing else. A tilted face still changes
the width of the beam in the plane of incidence, and still carries the
change of refractive index. Only in *reflection* is the uncurved plane left
with the identity.

The ray matrices are those of Siegman, *Lasers*, Table 15.1, with the
curvature given to one plane and zero to the other.
``tests/cymirror_verification.ipynb`` works through the comparison, and
``tests/gui/verify_cylindrical.py`` runs it as counted assertions.

.. _cylindrical-lens:

Cylindrical lens
~~~~~~~~~~~~~~~~

:py:class:`CyLens<gtrace.optcomp.CyLens>` is ordered exactly like a
:py:class:`Lens<gtrace.optcomp.Lens>`, with the same ``f``, the same
shapes, the same solver and the same refusals. It is shaped like a
:py:class:`CyMirror<gtrace.optcomp.CyMirror>`: both faces are cylinders
that share one ``curve_direction``. The focal length is therefore in that
plane, and the other plane is a plain window::

    from gtrace.optcomp import CyLens

    L = CyLens(f=500*mm)                       # focuses in the plane of the drawing
    L = CyLens(f=500*mm, curve_direction='v')  # focuses out of it

Everything said about cylindrical surfaces above applies here. A ``'v'``
lens is drawn as the rectangle that the plane of the trace cuts out of it.
Its focusing happens out of the page, carried by the ``qy`` of the beam.
The quoted ``f`` is the value at normal incidence. At a tilt, the two
planes scale differently.

The flat plane is a window, not nothing. A tilted flat face still rescales
the beam width in its plane of incidence, and the substrate is still a
length of glass.

Asking where the substrate is
------------------------------

These methods answer questions about the body of an element. They are not
about a beam meeting it.

:py:meth:`get_corners<gtrace.optcomp.Mirror.get_corners>` gives the four
corners, going round: the two ends of the HR chord, then the two ends of
the AR chord. The wedge is included, which is why the two sides come out
with different lengths. A curved face meets the sides at its chord, not at
its apex, and that is included too.

:py:meth:`get_side_info<gtrace.optcomp.Mirror.get_side_info>` describes the
sides as centre, normal and length, which is the form the hit test needs.
``get_corners`` gives the points instead, for code that has to point at the
element.

:py:meth:`contains_segment<gtrace.optcomp.Optics.contains_segment>` says
whether a straight span lies wholly inside the substrate::

    M.contains_segment(M.HRcenter, M.ARcenter)      # True: the optical thickness

It asks the optics itself, instead of describing its faces a second time.
:py:meth:`isHit<gtrace.optcomp.Mirror.isHit>` reports a surface only when
the ray approaches it from *outside*. It refuses any face that the ray is
leaving through. From inside a substrate it therefore finds nothing, in any
direction. That is all the test does. A face found partway along the span
means that the span starts outside. Ends that lie exactly on a face count
as inside, because that is where such a measurement is usually taken from.

The hollow of a concave face is the case to keep in mind. It is enclosed by
the substrate on three sides, and a test that only asked how far the
material reaches along a line would call it inside. It is air, and a span
across the front of a concave mirror is a span through air.
:py:meth:`contains_point<gtrace.optcomp.Optics.contains_point>` therefore
looks *both* ways from the point before deciding.

Optical systems
----------------

Mirrors and beams are enough on their own. The :doc:`tutorial` uses nothing
else to build its cavity. In a larger system, carrying lists of optics and
beams around becomes most of the code.
:py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` collects the optics,
the source beams and the tracing rules into one object. That object can be
traced, drawn, saved, and edited in the viewer. See :doc:`layout`.
