Propagation of a beam
===============================

A beam can be propagated either by manually propagate for a certain distance or tell gtrace to propagate it until hitting a particular mirror.

The manual propagation can be performed by calling, ``beam.propagate(d)`` where ``d`` is the distance to propagate.

Hitting a mirror
-----------------

.. image:: imgs/Mirror-beam-interaction.png
    :height: 12cm

By calling ``Mirror.hitFromHR(beam)``, you can tell gtrace to propagate the beam until it hits the mirror. If the beam indeed hits the mirror, gtrace will generate a set of beam objects produced by the interactions (reflection and refraction) of the incident beam with the mirror.

The generated beams are given the names indicated in the figure above.
The beam objects will be returned as a dictionary with the name of a beam as a key.

Non-sequential trace
---------------------

Following every beam by hand works for a handful of surfaces. A real layout produces far more ghost beams than that. :py:func:`non_seq_trace<gtrace.nonsequential.non_seq_trace>` follows them for you: give it a list of optics and one source beam, and it propagates every beam that is generated until a termination condition is met.

.. code-block:: python

    from gtrace.nonsequential import non_seq_trace

    beams = non_seq_trace([PRM, PR2, PR3], src,
                          order=10, power_threshold=1e-3)

A beam stops being followed when it hits nothing, when its power falls below ``power_threshold``, or when the number of internal reflections reaches ``order``. The return value is a flat list of the beams that were produced. Each carries a ``stray_order`` saying how many ghost reflections it took to make it, and the drawing uses that to tell the main beam from the ghosts.

If you keep your system in an :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>`, ``layout.trace()`` runs this for every registered source with the layout's rules. See :doc:`layout`.

.. _stray-order:

How the stray order is counted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every beam carries a counter, ``stray_order``. The source starts at zero, and the counter goes up by one each time the beam does something its element is not coated for. Whatever the element *is* coated for is free: reflecting off the front of a mirror costs nothing, and neither does the main beam's passage through a lens. What costs one order each is

- reflecting at the AR face, from either side and every time — an AR coating is not meant to reflect;
- passing through the HR face of a plain mirror — that coating is meant to send the beam back;
- reflecting at the HR face of an element whose front is not meant to reflect either — a lens.

So a ghost pays one order per round trip inside a substrate, counted at its AR bounce, and one more if it leaves through the HR. Leaving through the AR is free, since that is what an AR face is for. In a lens, whose faces are both meant to transmit, a bounce off either face counts.

A branch of the trace is dropped as soon as its counter exceeds ``order`` or its power falls below ``power_threshold``, whichever comes first. ``order`` bounds how *deep* a ghost may be, ``power_threshold`` how *faint*.

``order`` is a budget for the ghosts a call may **make**. It is not a test the arriving beam has to pass: a reflection off a face meant to reflect makes no ghost, so a beam that is already stray still bounces off a mirror, however deep it has got, and leaves at the order it arrived at.

**The counter rides with the beam.** It is not reset when a beam leaves one element for the next, so ``order`` is a budget for the whole trace. A ghost made at one mirror and steered by another is still that ghost: it is drawn as stray and carries its count with it. The counter is also what bounds the recursion, since each branch ends once it has spent the budget, so a system of mirrors facing each other terminates on ``order`` instead of waiting for the power to run out.

Which face is meant for what is said by two flags on the optics, both named from the HR side:

``HRtransmissive``
    The HR is meant to *transmit* as well as reflect, so the beam's first passage through it is free. False on a plain mirror. Set it True on a beam splitter or an input test mass. It defaults True on :py:class:`Lens<gtrace.optcomp.Lens>`, whose main beam goes through; with it False, the trace would count that main beam as a ghost and drop it at a low order.

``HRreflective``
    The mirror image: the HR is meant to *reflect*, so reflecting off it is free. True on a plain mirror, and still True on a beam splitter or an input test mass, where the reflection off the HR from inside the substrate is the interferometer's main return beam. It defaults False on :py:class:`Lens<gtrace.optcomp.Lens>` and :py:class:`CyLens<gtrace.optcomp.CyLens>`, where a reflection off a face is a ghost and is counted as one.

Two more attributes bound the unfolding rather than count it:

``max_stray_order``
    Caps the count for this one element, overriding the trace-wide ``order``. How deep the ghosts of one particular element are worth chasing is a property of that element.

``term_on_HR``
    Terminates a beam whose count is at most ``term_on_HR_order`` when it hits this HR, instead of reflecting it on. Without it, two facing high reflectors pass the main beam between them until the trace gives up. A cavity is the one configuration a non-sequential trace cannot finish on its own.
