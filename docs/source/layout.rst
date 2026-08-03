Optical layouts
===============================

An :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` holds a whole optical system in one object: the optics, the source beams and the rules that govern the non-sequential trace. It is what you draw, what you save to disk, and what the interactive viewer edits.

Nothing forces you to use it. The optics and beams described in :doc:`basic_concepts` and :doc:`propagation` work on their own, and the :doc:`tutorial` gets as far as a traced mode cleaner without one. What the layout adds is a single place to keep a system that you intend to trace, redraw and adjust repeatedly.

The two phases
---------------

A layout is not a substitute for the code that builds an optical system. Placing mirrors so that a cavity closes on itself, or so that a beam arrives at a mirror at the intended angle of incidence, is done with ordinary Python: sequential tracing, ``scipy.optimize``, cavity eigenmode solving. That code is where the physics of your design lives.

The intended workflow therefore has two phases:

**Phase 1 — build and align, with ordinary Python code.**

.. code-block:: python

    import gtrace.optcomp as opt
    import gtrace.beam as beam
    from gtrace.unit import *

    PRM = opt.Mirror(HRcenter=[0, 0], normAngleHR=0.0,
                     diameter=25*cm, thickness=10*cm,
                     inv_ROC_HR=1./458.1, name='PRM')
    PR2 = opt.Mirror(HRcenter=[14.7, 0], normAngleHR=deg2rad(180-2.5),
                     diameter=25*cm, thickness=10*cm, name='PR2')

    src = beam.GaussianBeam(q0=1j*10.0, pos=[-1.0, 0.0],
                            dirAngle=0.0, wl=1064*nm, name='src')

**Phase 2 — register the result and work with it as a whole.**

.. code-block:: python

    from gtrace.layout import OpticalLayout, TraceRules

    layout = OpticalLayout(optics=[PRM, PR2], sources=[src],
                           rules=TraceRules(order=10, power_threshold=1e-3),
                           name='PRC')

    layout.trace()          # non-sequential trace from every source
    layout.show()           # open the viewer

Held by reference
------------------

The layout does not copy what you register. ``layout.get_optics('PRM')`` returns the very object bound to ``PRM`` in your code. Consequently:

.. code-block:: python

    PRM.HRcenter = [0.01, 0.0]
    layout.trace()          # the new result reflects the move

and, in the other direction, an edit made in the viewer changes the object that ``PRM`` names. This is what makes the viewer more than a picture: there is exactly one copy of the model, and both you and the GUI are looking at it.

The same reasoning drives :py:meth:`update_from_file<gtrace.layout.OpticalLayout.update_from_file>`, which loads a saved layout *in place*: optics whose name and type match are updated rather than replaced, so ``PRM`` does not silently become a stale orphan the moment you load a file.

Note that :py:meth:`trace<gtrace.layout.OpticalLayout.trace>` is the one exception: it traces a *copy* of each source beam, so the registered source is not consumed by tracing and stays where you put it.

Tracing rules
--------------

:py:class:`TraceRules<gtrace.layout.TraceRules>` collects what governs the non-sequential trace:

``order``
    How many internal reflections are followed when a beam hits an element. Defaults to 10.

``power_threshold``
    Beams weaker than this are not propagated further. Defaults to 0.1. This is the knob to turn when chasing ghost beams: lowering it makes the trace find fainter paths, and slower.

``open_beam_length``
    How long a beam that hits nothing is drawn. Defaults to 1.0.

``order`` is a property of the trace as a whole. How deep the ghosts of *one particular element* are worth chasing is a property of that element, so it lives there instead, as the ``max_stray_order`` attribute of the optics.

Drawing options
----------------

How a layout is drawn is a display choice, not part of the model. Changing one redraws; it does not re-trace. The available options and their defaults are collected in ``gtrace.layout.DRAW_OPTIONS``:

============================ ========== =======================================
Option                       Default    Meaning
============================ ========== =======================================
``sigma_main``               2.7        Width of the drawn envelope of the main
                                        beams, in units of the 1/e² radius
``sigma_stray``              2.7        The same, for stray beams
``width_mode``               ``'x'``    Which transverse direction the envelope
                                        shows: ``'x'``, ``'y'`` or ``'avg'``
``drawMainWidth``            True       Whether to draw the main beam envelope
``drawStrayWidth``           True       Whether to draw the stray beam envelope
``drawBeamLabels``           False      Annotate each beam with its name and
                                        power
``drawOpticsNames``          True       Annotate each element with its name
``fontSize``                 False      Annotation size, or False for the
                                        gtrace default
============================ ========== =======================================

An option can be given per call, or set on the layout to apply to every subsequent drawing:

.. code-block:: python

    layout.render_html('trace.html', width_mode='y', sigma_main=1.0)

    layout.draw_options['width_mode'] = 'avg'   # applies from now on

A misspelt option raises ``TypeError`` rather than being ignored. A setting that is silently dropped looks exactly like a setting that had no effect, and there is no way to tell the two apart from the output.

.. _why-2.7-sigma:

Why 2.7 σ
^^^^^^^^^^

The envelope is drawn at 2.7 times the 1/e² radius because that is the aperture at which the diffraction loss of a Gaussian beam is 1 ppm. Drawing every envelope at the same σ means that every envelope in the picture carries the same meaning: "outside this line there is nothing that matters at the ppm level".

``width_mode`` exists because a beam is not round in general. After a beam passes through a wedged substrate at a non-normal incidence, its horizontal and vertical waists differ, and there is no single correct answer to "how wide is this beam". The drawing cannot make that choice for you.

Saving and loading
-------------------

A layout serialises to plain JSON:

.. code-block:: python

    layout.save('layout.json')

    layout2 = OpticalLayout.load('layout.json')     # a new layout
    layout.update_from_file('layout.json')          # in place, see above

The file holds the optics, the sources, the tracing rules and the drawing options. The trace result is not saved; it is regenerated by :py:meth:`trace<gtrace.layout.OpticalLayout.trace>`, and saving a derived quantity that can be recomputed only creates an opportunity for it to disagree with its inputs.

Editing from a front end
-------------------------

The viewer changes a layout by sending it messages, which you can also send yourself:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'PRM',
                       'HRcenter': [0.02, 0.0]})
    layout.apply_edit({'op': 'set', 'target': 'PRM',
                       'attrs': {'diameter': 0.15}})

The operations are ``move``, ``rotate``, ``set``, ``align``, ``slide``, ``add``, ``remove``, ``rename``, ``rules``, ``draw``, ``save``, ``load`` and ``undo``. Every message is a plain dict, so the same protocol travels over a notebook widget's comm as over any other transport.

A ``set`` may carry several attributes at once, and they are not applied in the order the message happens to list them: the anchor goes on before the curvatures it governs, and the orientation before the position that is measured from it. A message is a JSON object, whose key order is not something to rest on.

``align`` puts an element square across a beam, which is where almost every element on a bench is meant to sit and what a drag cannot say precisely. It names the beam by its index in the last trace, with the name as a check, and gives a point; the element is turned to face the beam and slid onto its axis at the projection of that point. See :doc:`viewer` for the Ctrl-drag that sends it.

``slide`` is the degree of freedom aligning leaves: it moves an element along a beam's axis by a distance in metres, positive downstream, and touches nothing else. It names the beam the same way::

    layout.apply_edit({'op': 'slide', 'target': 'L1',
                       'beam': 'b0', 'beam_index': 0, 'distance': 0.05})

The set of attributes a message may touch is an explicit whitelist (``EDITABLE_OPTIC_ATTRS``), and some attributes are further restricted to a set of permitted values (``ATTR_CHOICES``). Messages arrive from a browser, so "anything ``setattr`` accepts" is not a safe rule. An operation, target or attribute outside those sets raises :py:class:`EditError<gtrace.layout.EditError>` and leaves the layout untouched.

An attribute on the whitelist may still be one the target does not have, or one that refuses the value it is given. ``f`` is both: only a :py:class:`Lens<gtrace.optcomp.Lens>` has a focal length, and assigning to it re-solves both curvatures, which not every blank can be ground to. Either refusal comes back as an ``EditError`` with the reason, and the optics is left as it was.

``add`` builds a :py:class:`Mirror<gtrace.optcomp.Mirror>`, a :py:class:`CyMirror<gtrace.optcomp.CyMirror>` or a :py:class:`Lens<gtrace.optcomp.Lens>` (``CREATABLE_OPTIC_TYPES``). A mirror fills the parameters it was not given from the optics already registered, so that an element added to a system of 10 cm optics is a 10 cm optics. A lens does not: its coatings, aperture and wedge are the lens's own, and it is built from catalogue defaults at ``DEFAULT_LENS_F`` instead. It also accepts the parameters only a lens has (``CREATABLE_LENS_PARAMS``: ``f``, ``shape`` and ``ROC_HR``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                       'params': {'f': 0.3, 'shape': 'plano-convex',
                                  'HRcenter': [0.4, 0.0]}})

Renaming has its own operation rather than being an editable attribute, because the name is the identity that edits are resolved by; changing it needs a uniqueness check.

Two operations deliberately do *not* invalidate the trace result: ``draw`` changes display settings and ``save`` writes a file. Neither changes the physics, so neither causes a re-trace.

Undo
-----

Every edit that goes through ``apply_edit`` keeps the state before it, so it can be taken back:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
    layout.can_undo             # True
    layout.undo()               # or apply_edit({'op': 'undo'})

The history lives on the layout rather than in a front end, so undo means the same thing however the edit arrived. It goes back ``UNDO_DEPTH`` edits; there is no redo. A refused edit changes nothing and costs no step, and neither does ``save``, which only writes a file.

A step holds the registered elements themselves alongside their serialized values, and restoring one puts those values back onto those same objects — through a rename, and through a removal, since an element taken out of the layout is held in the history and put back as itself. See :py:meth:`undo<gtrace.layout.OpticalLayout.undo>` for what that does *not* cover.
