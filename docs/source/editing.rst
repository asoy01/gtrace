The edit protocol
===============================

The viewer changes a layout by sending it messages. This page is the
reference for those messages: what a message may contain, what it may
change, and what comes back when a message is refused. Read this page when
you want to edit a layout from code the way the viewer does, or when you
are writing a front end of your own. To use the viewer itself, read
:doc:`viewer`; to edit a layout from Python, read :doc:`layout`.

The message form
-----------------

Every message is a plain dict, passed to
:py:meth:`apply_edit<gtrace.layout.OpticalLayout.apply_edit>`. ``target``
is the name of an element, source, dimension or body registered in the
layout. The examples below use ``PRM``, the mirror of the layout built at
the top of :doc:`layout`:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'PRM',
                       'HRcenter': [0.02, 0.0]})
    layout.apply_edit({'op': 'set', 'target': 'PRM',
                       'attrs': {'diameter': 0.15}})

Because a message is a plain dict, the same protocol works over a notebook
widget's comm and over any other transport. There are seventeen
operations:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Operations
     - What they are for
   * - ``move``, ``rotate``, ``align``, ``slide``
     - Place the target.
   * - ``set``
     - Change the attributes of the target.
   * - ``add``, ``copy``, ``remove``, ``rename``
     - Change what the layout holds.
   * - ``rules``, ``draw``
     - Change the tracing rules and the drawing options.
   * - ``stretch``
     - Draw one beam that hits no surface at a length you choose.
   * - ``save``, ``load``, ``export``
     - Read and write files.
   * - ``undo``, ``redo``
     - Step through the history, which is described in :doc:`layout`.

Elements
---------

``move`` and ``rotate`` place an element, and ``set`` changes its
attributes. Two more operations do what a drag cannot do precisely.

``align`` puts an element perpendicular to a beam, which is where almost
every element on a bench is meant to sit. ``beam_index`` is the place of
the beam in the list the last trace made, ``beam`` is the name of that
beam and is checked against the index, and ``point`` is the point the
element is dropped at::

    layout.apply_edit({'op': 'align', 'target': 'L1',
                       'beam': 'b0', 'beam_index': 0,
                       'point': [0.4, 0.02]})

The element is turned to face the beam and slid onto the beam axis at the
projection of that point. See :doc:`viewer` for the Ctrl-drag that sends
this message.

``slide`` moves an element along the degree of freedom that ``align``
leaves free. It moves the element along the beam axis by a distance in
metres, positive downstream, and changes nothing else. The message names
the beam the same way::

    layout.apply_edit({'op': 'slide', 'target': 'L1',
                       'beam': 'b0', 'beam_index': 0, 'distance': 0.05})

``add`` builds a :py:class:`Mirror<gtrace.optcomp.Mirror>`, a
:py:class:`CyMirror<gtrace.optcomp.CyMirror>`, a
:py:class:`Lens<gtrace.optcomp.Lens>` or a
:py:class:`CyLens<gtrace.optcomp.CyLens>` (``CREATABLE_OPTIC_TYPES``). A
mirror takes the parameters it was not given from the optics registered
last: the diameter, the thickness, the wedge angle, the index and the four
coating values. A mirror added to a system of 10 cm optics is therefore a
10 cm mirror. Both faces are flat unless the message says otherwise. A lens
inherits none of those values. Its coatings, aperture and wedge are its
own, and it is built from catalogue defaults at ``DEFAULT_LENS_F``. ``add``
also accepts the parameters only a lens has (``CREATABLE_LENS_PARAMS``:
``f``, ``shape`` and ``ROC_HR``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                       'params': {'f': 0.3, 'shape': 'plano-convex',
                                  'HRcenter': [0.4, 0.0]}})

Renaming has its own operation instead of being an editable attribute.
Edits are resolved by name, so a rename needs a uniqueness check.

``copy`` adds a copy of an element, with the whole stack standing on it:
``{'op': 'copy', 'target': 'M1'}``. See :ref:`mechanics`.

.. _editing-a-source:

Sources
--------

The same operations reach the source beams, and do for a laser what they do
for an element: ``move`` sets where the laser stands, ``rotate`` sets which
way it fires, and ``set`` changes the beam it emits. ``b0`` below is the
name of a source registered in the layout.

.. code-block:: python

    layout.apply_edit({'op': 'move',   'target': 'b0', 'pos': [0.1, 0.0]})
    layout.apply_edit({'op': 'rotate', 'target': 'b0', 'dirAngle': 0.0})
    layout.apply_edit({'op': 'set',    'target': 'b0',
                       'attrs': {'waist_size_x': 0.35e-3,
                                 'waist_pos_x': 0.12}})
    layout.apply_edit({'op': 'add', 'type': 'Source', 'name': 'S1',
                       'params': {'pos': [0.0, 0.3],
                                  'waist_size': 0.2e-3}})

A source stands at a point and is aimed, so it has its own whitelist
(``EDITABLE_SOURCE_ATTRS``). ``move`` and ``rotate`` name ``pos`` and
``dirAngle``, instead of the centre and the face of an element. ``align``
and ``slide`` do not apply. There is no beam to put a laser perpendicular
to; the laser is where the beams start.

**A laser is specified by its waist, not by a q-parameter.**
``waist_size_x``, ``waist_size_y``, ``waist_pos_x`` and ``waist_pos_y`` are
not attributes a :py:class:`GaussianBeam<gtrace.beam.GaussianBeam>` has.
Each stands for one half of one q-parameter, and the edit protocol converts
it. Setting a size does not move the waist, moving the waist does not
change its size, and the two directions are independent:

.. code-block:: python

    b0 = layout.get_source('b0')

    def report():
        w = b0.waist()
        print('size %.3f / %.3f mm   at %.3f / %.3f m'
              % (w['Waist Size'][0]/mm, w['Waist Size'][1]/mm,
                 w['Waist Position'][0], w['Waist Position'][1]))

    report()
    layout.apply_edit({'op': 'set', 'target': 'b0',
                       'attrs': {'waist_size_x': 0.35e-3}})
    report()

::

    size 0.400 / 0.400 mm   at 0.000 / 0.000 m
    size 0.350 / 0.400 mm   at 0.000 / 0.000 m

The x waist changed size, the y waist did not, and neither one moved.

A waist position is the distance from the laser forward along the beam,
positive downstream, which is how
:py:meth:`waist<gtrace.beam.GaussianBeam.waist>` reports it. ``qx`` and
``qy`` may still be set directly, as ``[real, imag]``;
:py:func:`q_from_waist<gtrace.layout.q_from_waist>` and
:py:meth:`waist<gtrace.beam.GaussianBeam.waist>` convert between the two
descriptions.

**Through this protocol, changing the wavelength keeps the waist and
changes the divergence.** A q-parameter alone does not say how wide the
beam is; the width also depends on the wavelength. A change of wavelength
therefore has to keep either the waist or the q-parameter, and a laser is
specified by its waist.
:py:class:`GaussianBeam<gtrace.beam.GaussianBeam>` already works this way
for the refractive index: the change handler that runs when ``n`` is set
holds the reduced q fixed, and so keeps the waist size. Assigning
``b.wl`` in Python, outside this protocol, still keeps the q-parameter
instead.

A new source inherits nothing from the sources already registered, unlike a
new mirror. A laser is not built to match the laser beside it. A
q-parameter carried over would also be wrong: it describes a waist measured
from a point where the new source does not stand.
``DEFAULT_SOURCE_WAIST`` and ``DEFAULT_SOURCE_WL`` are
used instead. ``waist_size`` and ``waist_pos`` given to ``add`` stand for
both directions at once (``CREATABLE_SOURCE_PARAMS``).

**Optics, sources and dimensions share one namespace.** An edit message
names its target and nothing else. A name that meant one thing in one
message and another thing in the next message would be dangerous.
:py:meth:`add_source<gtrace.layout.OpticalLayout.add_source>` therefore
refuses a name an optics or a dimension has taken, as it has always done
for another source.

Beams
------

``stretch`` draws one beam longer or shorter than the trace made it. The
message has no ``target``. It names the beam by ``index``, the place of the
beam in the list the trace made, which is the order the ``beams`` channel
carries the beams in::

    layout.apply_edit({'op': 'stretch', 'index': 3, 'length': 2.5})

Only a beam that hits no surface can be stretched. Such a beam has ``open``
true in the ``beams`` channel, and the trace draws it at
``open_beam_length``, the length the tracing rules give every such beam. A
beam that ends on a surface is as long as the distance to that surface, and
``stretch`` on it is refused.

``stretch`` changes the drawing and nothing else. No beam is added, removed
or traced again. ``stretch`` is therefore not a step of undo, and the next
trace discards the length. Every beam also carries ``traced_length``, the
length the trace gave it, so a front end can restore that length. See
:ref:`drawing-a-beam-longer`.

Dimensions
-----------

A dimension is added and changed by the same operations, and it shares the
namespace described above. ``remove``, ``rename`` and ``set`` therefore
resolve their target across optics, sources, dimensions and bodies alike. A
message does not have to say which of the four its target is.

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                       'params': {'p1': list(M1.HRcenter),
                                  'p2': list(M1.ARcenter),
                                  'offset': 0.17}})
    layout.apply_edit({'op': 'set', 'target': 'D1',
                       'attrs': {'p2': [0.6, 0.0]}})

``move`` and ``rotate`` do not apply. A dimension is two points, not a
body, and either end moves on its own. What a dimension measures, and what
``offset`` does to the drawing, are described in :ref:`dimensions`.

Bodies
-------

The same operations reach a body, with its own whitelist
(``EDITABLE_MECHANICS_ATTRS``: ``center``, ``rotationAngle``,
``attached_to``, ``offset``, ``offset_angle``, ``fix_rotation``, ``width``
and ``height``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Mechanics', 'name': 'BB1',
                       'params': {'model': 'BB4530',
                                  'center': [0.3, 0.15]}})
    layout.apply_edit({'op': 'set', 'target': 'MT1',
                       'attrs': {'attached_to': 'M2'}})
    layout.apply_edit({'op': 'set', 'target': 'BB1',
                       'attrs': {'width': 0.6, 'height': 0.45}})

An ``add`` naming a ``model`` and no ``shapes`` builds the body from the
library; an ``add`` carrying ``shapes`` builds it from those shapes.
``attached_to`` takes the name of an optics **or of another body**, or
``None``. Seating a body on an *optics* puts it at the model's place, since
the library, not the cursor, decides where a mount belongs on a mirror.
Seating a body on another *body* keeps it where it already is, since which
hole of a mount a pedestal sits in is a choice made on the bench. Setting
``width`` or ``height`` goes through ``resize``, which reports an error
when the body has no size to set. A ``rotate`` on an attached body is
refused unless the body is free to turn.

Attaching through this protocol takes the attach point from the drawing.
The point of the body that already coincides with a point of the host is
the point the body is pinned by. Dropping a body on a hole and then
attaching it therefore pins the body by that hole. See :ref:`mechanics`.

An assembly and a beam dump are each one ``add``, so each is one step of
undo::

    {'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-2IN',
     'params': {'center': [0.3, 0.1], 'angle': 0.7854}}

    {'op': 'add', 'type': 'BeamDump', 'name': 'BD1',
     'params': {'center': [0.3, 0.0], 'angle': 0.0}}

Neither an assembly nor a beam dump is a model in the library, and neither
can be. A model holds shapes only, and the first piece of an assembly or of
a dump is an element. On a dump, a front end may set ``center``, ``angle``
and ``reflectivity``. The other attributes come from the drawing.

Shapes
-------

:py:class:`ShapeEditor<gtrace.draw.viewer.editor.ShapeEditor>` is the model
behind the shape editor. It can be used without a browser, and it has a
protocol of its own:

.. code-block:: python

    from gtrace.draw.viewer.editor import ShapeEditor

    ed = ShapeEditor(clamp)
    ed.apply_edit({'op': 'add_shape', 'type': 'circle'})
    ed.apply_edit({'op': 'set_shape', 'index': 2,
                   'attrs': {'radius': 0.004}})
    ed.apply_edit({'op': 'rotate_shape', 'index': 0, 'angle': 0.7854})
    ed.apply_edit({'op': 'set_points',
                   'points': [{'name': 'post', 'point': [-0.0135, 0.0]}]})
    ed.apply_edit({'op': 'undo'})

``add_shape`` takes the ``params`` of the shape, which is how the viewer
sends a shape the user has drawn by clicking: the clicked points become a
``start`` and a ``stop``, a ``center`` and a ``radius``, and so on. With no
``params``, ``add_shape`` puts down the default shape of that kind, which
is what ``newshapes`` carries.

The operations are ``add_shape``, ``set_shape``, ``remove_shape``,
``duplicate_shape``, ``move_shape``, ``rotate_shape``, ``set_points``,
``save_model``, ``undo`` and ``redo``. A shape is edited in three steps:
convert the shape into the dict that
:py:func:`shape_to_dict<gtrace.draw.serialize.shape_to_dict>` writes,
change the values the message names, and build the shape again. The
constructors are therefore the only rule about what a shape is. A few
values the constructors do not reject are refused afterwards: a size of
zero or less, a coordinate at infinity, and an outline with one vertex. An
index is a **place in the list**, which is also the order the shapes are
drawn in, so removing one shape renumbers the shapes after it.

A turn is the one edit that is not a set of attributes, because turning
means something different for each kind of shape. The two angles of an arc
move. A text turns with its own rotation. A ``Rectangle`` carries a turn of
its own, an ``angle`` and the ``pivot`` it is taken about, so it stores
those two values and stays a rectangle. The rectangle keeps a width and a
height that you can still edit. Every other kind goes through
:py:func:`turned_shape<gtrace.mechanics.turned_shape>`. ``pivot`` defaults
to :py:func:`shape_centre<gtrace.mechanics.shape_centre>`, the middle of the
bounding box of the shape.

The turn of a **body** is a different matter, and it is not written into
the shapes. The pose of a body says where it stands and which way it faces,
and the shapes are read in the frame of the body. A rectangle carried by a
turned body therefore still appears on the bench as the closed polyline of
its four corners. A DXF file holds that polyline in either case.

``set_points`` carries the **whole list** of named points, not one point.
An index does not survive a rename: a point is known by its name, and the
name itself is what an edit may change. Renaming, moving, adding and
removing a point are all the same message, so each of them is one step of
undo. Two points cannot share a name, and a point cannot be unnamed. The
scene channel is ``points``, a list of
``{'name': str, 'point': [x, y], 'index': int}``.

The editor holds the ``Mechanics`` **by reference**, like everything else
in this protocol. A body already registered in a layout is therefore
redrawn at the next draw of that layout. The attachment, the pose and the
builder parameters of the body are unchanged.

Rules, drawing and files
-------------------------

The tracing rules have their own operation, and each value is checked.
``order`` is a whole number no greater than ``MAX_RULE_ORDER``, since each
order is another round of reflections at every element::

    layout.apply_edit({'op': 'rules', 'rules': {'order': 20,
                                                'power_threshold': 1e-9}})

``draw`` changes the drawing options of the layout. Its ``params`` have a
whitelist of their own (``EDITABLE_DRAW_OPTIONS``: ``sigma_main``,
``sigma_stray``, ``width_mode``, ``drawMainWidth``, ``drawStrayWidth``,
``drawBeamLabels``, ``drawOpticsNames`` and ``drawMechanicsNames``)::

    layout.apply_edit({'op': 'draw', 'params': {'sigma_main': 1.0,
                                                'width_mode': 'y'}})

``save`` writes the layout to a JSON file, and ``load`` reads one back::

    layout.apply_edit({'op': 'save', 'path': 'layout.json'})
    layout.apply_edit({'op': 'load', 'path': 'layout.json'})

``load`` calls
:py:meth:`update_from_file<gtrace.layout.OpticalLayout.update_from_file>`,
so it fills this layout object instead of returning a new one. An element
of the file that matches a registered element by name and by class is
updated in place, so a variable that holds the element keeps pointing at
the right object. Everything else is built afresh, and a registered
element the file does not name is dropped. ``load`` is one step of undo.

Four operations do *not* invalidate the trace result: ``draw`` changes
display settings, ``save`` and ``export`` write a file, and ``stretch``
changes how far an open beam is drawn. None of them changes the physics, so
none causes a re-trace. An edit to a dimension does not cause a re-trace
either.

``export`` writes the drawing, not the model. Today the only export is
``{'op': 'export', 'format': 'dxf', 'path': ...}``, which calls
:py:meth:`export_dxf<gtrace.layout.OpticalLayout.export_dxf>`. See
:ref:`dxf-export`.

What a message may change
--------------------------

The set of attributes a message may change is an explicit whitelist
(``EDITABLE_OPTIC_ATTRS``), and some attributes are further restricted to a
set of permitted values (``ATTR_CHOICES``). An operation, target or
attribute outside those sets raises
:py:class:`EditError<gtrace.layout.EditError>` and leaves the layout
untouched.

An attribute on the whitelist may still be one the target does not have, or
one that refuses the value it is given. Either refusal comes back as an
``EditError`` with the reason, and the optics is left as it was.

``f`` is both kinds at once. Only a :py:class:`Lens<gtrace.optcomp.Lens>`
has a focal length, so ``f`` on a mirror is refused. Assigning to ``f``
scales both curvatures together until the lens has that focal length, and
the scaling can fail in two ways.
Sometimes no scaling of the shape the lens already has reaches the focal
length asked for. Sometimes the curvatures that do reach it cannot be
ground from the blank: a face would be steeper than its own aperture, or
the two concave faces would meet in the middle.

A ``set`` may carry several attributes at once. The attributes are not
applied in the order the message lists them. The anchor is applied before
the curvatures it governs, and the orientation before the position that is
measured from that orientation. A message is a JSON object, so you cannot
rely on the order of its keys.

Scene channels
---------------

:py:meth:`scene_dict<gtrace.layout.OpticalLayout.scene_dict>` adds ten
entries to what
:py:func:`scene_to_dict<gtrace.draw.serialize.scene_to_dict>` builds:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Channel
     - What it carries
   * - ``can_undo``
     - Whether the front end's Undo has anything to work with.
   * - ``can_redo``
     - Whether the front end's Redo has anything to work with.
   * - ``dimensions``
     - The dimensions, with their measurements.
   * - ``snap``
     - The points a front end may snap a measurement to.
   * - ``sources``
     - The laser sources.
   * - ``rules``
     - The tracing rules.
   * - ``mechanics``
     - The bodies that are registered in the layout.
   * - ``mechlib``
     - The model library a new body can be built from.
   * - ``assemblies``
     - The assemblies a front end can offer.
   * - ``newshapes``
     - What a new shape of each kind looks like.

``sources`` says which of the beams the user put there. No other part of
the scene carries that information. A source is traced from a *copy* of
itself, so its own beam sits in ``beams`` and looks like the beams the
trace made from it. Each entry carries where the laser stands, which way it
fires, and the beam it emits, including the waist. The waist is computed on
the Python side and not stored, for the same reason the length of a
dimension is.
``rules`` carries the tracing rules. The rules belong to no element, but
they decide how much of the picture there is.

Each dimension carries a ``line``: the two ends the line lands on once the
offset is applied. :py:meth:`line_ends<gtrace.layout.Dimension.line_ends>`
works those two ends out on the Python side, so a front end does not have
to decide which side of the two points the offset goes to.

``mechanics`` carries the pose of each body, what it is attached to, and
the outline a front end picks it by. A body that is one shape drawn by hand
also carries that ``shape``, in the frame the body is written in. Such a
body is a drawing, not a part, so you edit the numbers of the shape itself:
``{'op': 'set', 'target': ..., 'attrs': {'shape': {...}}}`` sets them,
through the same rules the shape editor applies. A part from the library is
cut to size with ``width`` and ``height`` instead. A body of several shapes
is edited with :py:meth:`edit<gtrace.mechanics.Mechanics.edit>`. Both
refuse a ``shape``, instead of guessing which shape was meant. Python
computes the outline. It is the same polygon
:py:meth:`contains<gtrace.mechanics.Mechanics.contains>` tests against, so
a browser does not need a second description of it.

``mechlib`` is the model library, as names, descriptions and name prefixes.
The ``+ Mechanics`` menu shows those names, and it uses the prefixes to
name the bodies it adds. The shapes stay on the Python side until a model
is chosen. ``assemblies`` is what
:py:func:`assembly_kinds<gtrace.layout.assembly_kinds>` lists, so a front
end can offer an element together with the parts that hold it, by name.
Python builds the assemblies. Each kind carries a ``place``, which names
the parameter that says where it goes: ``HRcenter`` for a mirror,
``center`` for a lens. A front end that has one clicked point sends it
under that name.

``newshapes`` says what a shape of each kind looks like when it is first
put down. It is the same
:py:data:`NEW_SHAPES<gtrace.draw.serialize.NEW_SHAPES>` that a shape editor
uses. ``+ Shape`` can therefore add a body of one shape, and a front end
does not need its own size for a new circle. The sizes are bench sizes. A
front end that shows kilometres has to scale them to what it shows.

``snap`` carries, for each substrate, its four corners, the apex of each
face, its middle, and the middle of each of its two sides. It also carries,
for each body, the four corners of its outline, the middle of each of those
four edges, the points the body names for itself, and the centre of every
screw hole it has.

Each point says what ``kind`` it is. A front end that offers points for one
purpose and not another reads that ``kind``: the viewer takes every kind
when measuring and aiming, and leaves out ``midpoint`` when a part is
dragged onto another. The middle of an edge is a place to measure from, not
a place to fix a part to.

Only straight edges get a middle. The middle of a curved face is its apex,
which is already on the list; the middle of its chord is inside the glass,
where nothing is drawn.

The named points come before the holes. Two marks at the same place count
as one point, and the first one is kept. The post hole of a mount is both a
circle in the drawing and the point where the mount stands on its pedestal,
and ``MT post`` is a more useful label than ``MT hole``.

These points come from Python because they are geometry: a corner is where
the wedge and the sagitta of a curved face put it. Beam ends are
deliberately *not* in ``snap``. The scene already carries the ends of every
beam, so a front end can offer those directly.
