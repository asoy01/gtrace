Propagation of a beam
======================

There are three ways to move a beam, from the most manual to the most
automatic:

1. advance the beam yourself by a distance;
2. give the beam to one element and collect the beams that come off it;
3. let gtrace follow every beam through the whole system.

A layout does the third one, and it uses the first two to do it.

Moving a beam by hand
----------------------

:py:meth:`propagate(d)<gtrace.beam.GaussianBeam.propagate>` moves a beam
``d`` metres along its own direction:

.. code-block:: python

    from gtrace.beam import GaussianBeam
    from gtrace.layout import q_from_waist
    from gtrace.unit import *

    b = GaussianBeam(q0=q_from_waist(0.4*mm, 0.0, 1064*nm), wl=1064*nm,
                     pos=[0.0, 0.0], dirAngle=0.0, name='b')
    b.propagate(1.0)

This moves ``pos``, advances the q parameter, and adds to the accumulated
optical distance ``optDist`` and Gouy phase ``Gouyx`` and ``Gouyy``. It
changes the beam in place, so call :py:meth:`copy()<gtrace.beam.GaussianBeam.copy>`
first if you need the old one as well.

To ask what a beam will be doing further on without moving it, use
:py:meth:`width(d)<gtrace.beam.GaussianBeam.width>`, which returns the
radius the beam would have ``d`` metres ahead::

    >>> b.width(1.0)                     # (wx, wy), in metres
    (0.0009364337493809829, 0.0009364337493809829)

Hitting a mirror
-----------------

.. image:: imgs/Mirror-beam-interaction.png
    :height: 12cm

:py:meth:`hitFromHR(beam)<gtrace.optcomp.Mirror.hitFromHR>` propagates the
beam until it reaches the mirror's HR face, then computes the reflections
and refractions there and inside the substrate.
:py:meth:`hitFromAR<gtrace.optcomp.Mirror.hitFromAR>` does the same from
the back face. Both return a dictionary of beams, keyed by the names in
the figure::

    >>> beams = M.hitFromHR(src, order=2)
    >>> sorted(beams)
    ['input', 'r1', 's1', 's2', 's3', 't1', 't2']

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Beam
   * - ``input``
     - The incident beam, cut off at the surface it hit.
   * - ``r1``
     - Reflected off the HR face, back the way it came.
   * - ``s1``
     - Refracted into the substrate, on its way to the AR face.
   * - ``t1``
     - Transmitted through both faces and out the far side.
   * - ``s2``, ``s3``, …
     - Successive bounces inside the substrate.
   * - ``r2``, ``t2``, …
     - What leaves after each of those bounces, through the HR and the
       AR face respectively.

The powers say which of them are worth following. For a mirror of 90%
reflectivity with a 0.5% AR coating, hit by a 1 W beam:

::

    r1   0.9        W
    s1   0.1        W
    t1   0.0995     W
    s2   0.0005     W
    s3   0.00045    W
    t2   0.0004478  W

``order`` says how many round trips inside the substrate to follow, and
``threshold`` drops beams below a given power. A beam that misses the
mirror altogether gets an empty dictionary back.

Tracing a whole system
-----------------------

Calling ``hitFromHR`` by hand works for a few surfaces. A real layout
produces many more beams than that. You do not want to work out which
element each beam reaches next.
:py:func:`non_seq_trace<gtrace.nonsequential.non_seq_trace>` does that
work for you. Give it a list of optics and one source beam. It follows
every beam that is generated, until the beam stops.

.. code-block:: python

    from gtrace.nonsequential import non_seq_trace

    beams = non_seq_trace([PRM, PR2, PR3], src,
                          order=10, power_threshold=1e-3)

A beam stops being followed when it hits nothing, when its power falls
below ``power_threshold``, or when its stray order reaches ``order``. The
return value is a flat list of every beam produced. Each one carries a
``stray_order`` saying how many ghost reflections it took to make it, and
the drawing uses that to tell the main beam from the ghosts.

:py:meth:`layout.trace()<gtrace.layout.OpticalLayout.trace>` runs this for
every source registered in an
:py:class:`OpticalLayout<gtrace.layout.OpticalLayout>`, with the layout's
own rules. See :doc:`layout`.

.. _stray-order:

Ghost beams and the stray order
--------------------------------

Every beam carries a counter, ``stray_order``. A source beam starts at
zero, and the counter goes up by one each time the beam does something its
element is not coated for. What the element *is* coated for costs nothing:
reflecting off the front of a mirror is free, and so is the main beam's
passage through a lens. What costs one order each is

- reflecting at the AR face, from either side and every time. An AR
  coating is not meant to reflect.
- passing through the HR face of a plain mirror. That coating is meant to
  send the beam back.
- reflecting at the HR face of an element whose front is not meant to
  reflect either, such as a lens.

A ghost therefore pays one order per round trip inside a substrate,
counted at its AR bounce, and one more if it leaves through the HR.
Leaving through the AR is free. In a lens, where both faces are meant to
transmit, a bounce off either face costs one.

A branch of the trace is dropped as soon as its counter exceeds ``order``
or its power falls below ``power_threshold``, whichever comes first.
``order`` limits how *deep* a ghost may be, ``power_threshold`` how
*faint*.

``order`` is a budget for the ghosts a call may **make**. The arriving beam
is not tested against it. A reflection off a face that is meant to reflect
makes no ghost. A beam that is already stray therefore still bounces off a
mirror, no matter how deep it has got. It leaves with the order it arrived
with.

**The counter travels with the beam.** It is not reset when a beam leaves
one element for the next, so ``order`` is a budget for the whole trace. A
ghost made at one mirror and steered by another is still that ghost: it is
drawn as stray and keeps its count. The counter is also what bounds the
recursion, since each branch ends once it has spent the budget. Two mirrors
facing each other therefore terminate on ``order`` instead of running until
the power runs out.

Which face is meant for what
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two flags on the optics say it, both named from the HR side:

``HRtransmissive``
    The HR is meant to *transmit* as well as reflect, so the beam's first
    passage through it is free. False on a plain mirror. Set it True on a
    beam splitter or an input test mass. It defaults to True on
    :py:class:`Lens<gtrace.optcomp.Lens>`, whose main beam goes through;
    with it False, the trace would count that main beam as a ghost and drop
    it at a low order.

``HRreflective``
    The opposite: the HR is meant to *reflect*, so reflecting off it is
    free. True on a plain mirror. It stays True on a beam splitter or an
    input test mass. On those elements, the reflection off the HR from
    inside the substrate is the main return beam of the interferometer.
    It defaults to False
    on :py:class:`Lens<gtrace.optcomp.Lens>` and
    :py:class:`CyLens<gtrace.optcomp.CyLens>`, where a reflection off a face
    is a ghost and is counted as one.

Stopping a beam at a surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two more attributes stop the unfolding, instead of counting it:

``max_stray_order``
    Caps the count for this one element, overriding the trace-wide
    ``order``. How deep the ghosts of one particular element should be
    followed is a property of that element.

``term_on_HR``
    Ends a beam whose count is at most ``term_on_HR_order`` when it reaches
    this HR face, instead of reflecting it on. Without it, two facing high
    reflectors pass the main beam between them until the trace gives up. A
    non-sequential trace cannot finish a cavity on its own, so you have to
    tell it where to stop.

``term_on_HR_transmits``
    Says how much ``term_on_HR`` stops. False is the default. It stops the
    beam at the surface and computes nothing there. True stops only the
    reflection that would come back. The element is hit as usual otherwise,
    so the beam transmitted through the substrate carries on and is drawn.
    That is the beam a detector behind a cavity mirror sees.

    The beams that survive go through the same code as any other beam.
    ``order``, ``max_stray_order`` and the power threshold therefore count
    them and cap them. Only the external reflection is dropped. A ghost
    that leaves through the HR from inside the substrate is still a ghost,
    and the budget decides how far it goes. The gate itself is unchanged,
    so a beam above ``term_on_HR_order`` reflects as before.
