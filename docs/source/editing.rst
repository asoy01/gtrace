The edit protocol
===============================

The viewer changes a layout by sending it messages. This page is the
reference for those messages: what they may say, what they may touch, and
what comes back when one is refused. You do not need it to use the viewer,
which is described in :doc:`viewer`, nor to use a layout from Python, which
is :doc:`layout`. Read it when you want to drive a layout from code the way
the viewer does, or when you are writing a front end of your own.

The message form
-----------------

Every message is a plain dict, handed to
:py:meth:`apply_edit<gtrace.layout.OpticalLayout.apply_edit>`:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'PRM',
                       'HRcenter': [0.02, 0.0]})
    layout.apply_edit({'op': 'set', 'target': 'PRM',
                       'attrs': {'diameter': 0.15}})

The operations are ``move``, ``rotate``, ``set``, ``align``, ``slide``,
``add``, ``copy``, ``remove``, ``rename``, ``rules``, ``draw``, ``save``,
``load``, ``export``, ``undo`` and ``redo``. Because a message is a plain
dict, the same protocol travels over a notebook widget's comm as over any
other transport.

The set of attributes a message may touch is an explicit whitelist
(``EDITABLE_OPTIC_ATTRS``), and some attributes are further restricted to a
set of permitted values (``ATTR_CHOICES``). Messages arrive from a browser,
so "anything ``setattr`` accepts" is not a safe rule. An operation, target
or attribute outside those sets raises
:py:class:`EditError<gtrace.layout.EditError>` and leaves the layout
untouched.

An attribute on the whitelist may still be one the target does not have, or
one that refuses the value it is given. ``f`` is both: only a
:py:class:`Lens<gtrace.optcomp.Lens>` has a focal length, and assigning to
it re-solves both curvatures, which not every blank can be ground to.
Either refusal comes back as an ``EditError`` with the reason, and the
optics is left as it was.

A ``set`` may carry several attributes at once. They are not applied in the
order the message lists them. The anchor is applied before the curvatures
it governs, and the orientation before the position that is measured from
it. A message is a JSON object, so you cannot rely on the order of its
keys.

Elements
---------

``move`` and ``rotate`` place an element, and ``set`` changes its
attributes. Two more operations cover what a drag cannot say precisely.

``align`` puts an element square across a beam, which is where almost every
element on a bench is meant to sit. It names the beam by its index in the
last trace, with the name as a check, and gives a point. The element is
turned to face the beam and slid onto its axis at the projection of that
point. See :doc:`viewer` for the Ctrl-drag that sends it.

``slide`` is the degree of freedom aligning leaves. It moves an element
along a beam's axis by a distance in metres, positive downstream, and
touches nothing else. It names the beam the same way::

    layout.apply_edit({'op': 'slide', 'target': 'L1',
                       'beam': 'b0', 'beam_index': 0, 'distance': 0.05})

``add`` builds a :py:class:`Mirror<gtrace.optcomp.Mirror>`, a
:py:class:`CyMirror<gtrace.optcomp.CyMirror>` or a
:py:class:`Lens<gtrace.optcomp.Lens>` (``CREATABLE_OPTIC_TYPES``). A mirror
takes the parameters it was not given from the optics already registered.
An element added to a system of 10 cm optics is therefore a 10 cm optics. A
lens does not do this. Its coatings, aperture and wedge are its own, and it
is built from catalogue defaults at ``DEFAULT_LENS_F``. It also accepts the
parameters only a lens has (``CREATABLE_LENS_PARAMS``: ``f``, ``shape`` and
``ROC_HR``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                       'params': {'f': 0.3, 'shape': 'plano-convex',
                                  'HRcenter': [0.4, 0.0]}})

Renaming has its own operation instead of being an editable attribute. The
name is the identity that edits are resolved by, so changing it needs a
uniqueness check.

``copy`` — ``{'op': 'copy', 'target': 'M1'}`` — adds a second one of an
element with the whole stack standing on it. See :ref:`mechanics`.

.. _editing-a-source:

Sources
--------

The same operations reach the source beams, and mean for a laser what they
mean for an element: ``move`` says where it stands, ``rotate`` which way it
fires, ``set`` what light it puts out.

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
and ``slide`` do not apply. There is no beam to put a laser square across;
the laser is where the beams start.

**A laser is specified by its waist, not by a q-parameter.**
``waist_size_x``, ``waist_size_y``, ``waist_pos_x`` and ``waist_pos_y`` are
not attributes a :py:class:`GaussianBeam<gtrace.beam.GaussianBeam>` has.
Each stands for one half of one q-parameter and is converted here. Setting a
size does not move the waist, moving it does not resize it, and the two
directions are independent. ``qx`` and ``qy`` may still be set directly, as
``[real, imag]``; :py:func:`q_from_waist<gtrace.layout.q_from_waist>` and
:py:meth:`waist<gtrace.beam.GaussianBeam.waist>` convert between the two
descriptions.

A waist position is the distance from the laser forward along the beam,
positive downstream, which is how
:py:meth:`waist<gtrace.beam.GaussianBeam.waist>` reports it.

**Through this protocol, changing the wavelength keeps the waist and
changes the divergence.** A q-parameter alone does not say how wide the
beam is; the width also depends on the wavelength. Changing one of the two
therefore has to keep the other. The waist is what a laser is specified by.
The model already works this way for the refractive index: that handler
holds the reduced q fixed. Assigning ``b.wl`` directly in a cell is
unchanged, and keeps the q-parameter instead.

A new source inherits nothing from the sources already registered, and a
new mirror does. A laser is not cut to match the one beside it. A
q-parameter carried over would also be wrong: it describes a waist measured
from a point where the new source does not stand.
``DEFAULT_SOURCE_WAIST`` and ``DEFAULT_SOURCE_WL`` are
used instead. ``waist_size`` and ``waist_pos`` given to ``add`` stand for
both directions at once (``CREATABLE_SOURCE_PARAMS``).

**Optics, sources and dimensions share one namespace.** An edit message
names its target and nothing else. A name that meant one thing in one
message and another thing in the next message would be dangerous.
:py:meth:`add_source<gtrace.layout.OpticalLayout.add_source>` therefore
refuses a name an optics or a dimension has taken, as it always did for
another source.

Dimensions
-----------

A dimension is added and changed by the same operations, and it shares the
namespace above. ``remove``, ``rename`` and ``set`` therefore resolve their
target across optics and dimensions alike. A front end has a name under the
cursor, not a class.

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                       'params': {'p1': list(M1.HRcenter),
                                  'p2': list(M1.ARcenter),
                                  'offset': 0.17}})
    layout.apply_edit({'op': 'set', 'target': 'D1',
                       'attrs': {'p2': [0.6, 0.0]}})

``move`` and ``rotate`` do not apply. A dimension is two points, not a
body, and either end moves on its own. What a dimension measures, and what
``offset`` does to the drawing, are in :ref:`dimensions`.

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

An ``add`` naming a ``model`` and no ``shapes`` is built from the library;
one carrying ``shapes`` is built from them. ``attached_to`` takes the name
of an optics **or of another body**, or ``None``. Seating a body on an
*optics* puts it at the model's place, since where a mount belongs on a
mirror is the library's business and not the cursor's. Seating it on a
*body* keeps where it already is, since which hole of a mount a pedestal
sits in is a choice made on the bench. Setting ``width`` or ``height`` goes
through ``resize``, which says so when the body has no size to set. A
``rotate`` on an attached body is refused unless its turn is free.

Attaching through this protocol takes the attach point from the drawing.
The point of the body that already coincides with a point of the host is
the point the body is pinned by. So "drop it on the hole, then attach it"
pins the body by that hole. See :ref:`mechanics`.

An assembly and a beam dump are each one ``add``, so each is one step of
undo::

    {'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-2IN',
     'params': {'center': [0.3, 0.1], 'angle': 0.7854}}

    {'op': 'add', 'type': 'BeamDump', 'name': 'BD1',
     'params': {'center': [0.3, 0.0], 'angle': 0.0}}

Neither one is a model in the library, and neither one can be. A model
holds shapes only, and the first piece of an assembly or of a dump is an
element. On a dump, a front end may set ``center``, ``angle`` and
``reflectivity``. The rest comes from the drawing.

Shapes
-------

:py:class:`ShapeEditor<gtrace.draw.viewer.editor.ShapeEditor>` is the model
behind the shape editor. It is drivable without a browser and speaks a
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

The operations are ``add_shape``, ``set_shape``, ``remove_shape``,
``duplicate_shape``, ``move_shape``, ``rotate_shape``, ``set_points``,
``save_model``, ``undo`` and ``redo``. A shape is edited in three steps:
take it apart into the dict that
:py:func:`shape_to_dict<gtrace.draw.serialize.shape_to_dict>` writes,
change what the message names, and build the shape again. The constructors
are therefore the only rule about what a shape is. A few things they do not
catch are refused on the way out: a size of zero or less, a coordinate at
infinity, and an outline with one vertex. An index is a **place in the
list**, which is also the order the shapes are drawn in, so removing one
shape renumbers the shapes after it.

A turn is the one edit that is not a set of attributes, because turning
means something different for each kind of shape. The two angles of an arc
move. A text turns with its own rotation. A ``Rectangle`` carries a turn of
its own, an ``angle`` and the ``pivot`` it is taken about, so it stores
those and stays a rectangle. It keeps a width and a height you can go on
editing. Every other kind goes through
:py:func:`turned_shape<gtrace.mechanics.turned_shape>`. ``pivot`` defaults
to :py:func:`shape_centre<gtrace.mechanics.shape_centre>`, the middle of the
bounding box of the shape.

The turn of a **body** is a different question, and it is not written into
the shapes. The pose of a body says where it stands and which way it faces,
and the shapes are read in the frame of the body. A rectangle carried by a
turned body therefore still reaches the bench as the closed polyline of its
four corners. That is what a DXF file holds in either case.

``set_points`` carries the **whole list** of named points, not one point.
No index survives a rename: a point is known by its name, and the name is
the thing being edited. Renaming a point, moving one, adding one and taking
one away are all the same message, which also makes each of them one step
of undo. Two points cannot share a name, and a point cannot go unnamed. The
scene channel is ``points``, a list of
``{'name': str, 'point': [x, y], 'index': int}``.

The editor holds the ``Mechanics`` **by reference**, like everything else
here. A body already registered in a layout is therefore redrawn at the
next draw of that layout. Its attachment, its pose and its builder
parameters are untouched.

Rules, drawing and files
-------------------------

The tracing rules have their own operation, and each value is checked.
``order`` is a whole number no greater than ``MAX_RULE_ORDER``, since each
order is another round of reflections at every element::

    layout.apply_edit({'op': 'rules', 'rules': {'order': 20,
                                                'power_threshold': 1e-9}})

Three operations do *not* invalidate the trace result: ``draw`` changes
display settings, and ``save`` and ``export`` write a file. None of them
changes the physics, so none causes a re-trace. Nor does anything done to a
dimension.

``export`` writes the drawing, not the model. Today that is only
``{'op': 'export', 'format': 'dxf', 'path': ...}``, which is
:py:meth:`export_dxf<gtrace.layout.OpticalLayout.export_dxf>`. See
:ref:`dxf-export`.

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
   * - ``can_undo``, ``can_redo``
     - Whether the front end's Undo and Redo have anything to work with.
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

``sources`` says which of the beams the user put there. Nothing else in the
scene can say it. A source is traced from a *copy* of itself, so its own
beam sits in ``beams`` and looks like the beams the trace made from it.
Each entry carries where the laser stands, which way it fires, and the
light it emits. The waist is included. It is computed here and not stored,
for the same reason the length of a dimension is. ``rules`` carries the
tracing rules. They belong to no element, but they decide how much of the
picture there is.

Each dimension carries a ``line``: the two ends the line lands on once the
offset is applied. Only one place therefore decides which side the offset
goes to.

``mechanics`` carries the pose of each body, what it is attached to, and
the outline a front end picks it by. A body that is one shape drawn by hand
also carries that ``shape``, in the frame the body is written in. Such a
body is a drawing, not a part, so its own numbers are what you edit:
``{'op': 'set', 'target': ..., 'attrs': {'shape': {...}}}`` sets them,
through the same rules the shape editor applies. A part from the library is
cut to size with ``width`` and ``height`` instead. A body of several shapes
is edited with :py:meth:`edit<gtrace.mechanics.Mechanics.edit>`. Both
refuse a ``shape``, instead of guessing which shape was meant. The outline
is computed here. It is the same polygon
:py:meth:`contains<gtrace.mechanics.Mechanics.contains>` tests against, and
a browser has no reason to hold a second description of it.

``mechlib`` is the model library, as names, descriptions and name prefixes.
The ``+ Mechanics`` menu shows those names, and it uses the prefixes to
name the bodies it adds. The shapes stay on the Python side until a model
is chosen. ``assemblies`` is what
:py:func:`assembly_kinds<gtrace.layout.assembly_kinds>` lists, so a front
end can offer an element together with the parts that hold it, by name.
Python builds them.

``newshapes`` says what a shape of each kind looks like when it is first
put down. It is the same
:py:data:`NEW_SHAPES<gtrace.draw.serialize.NEW_SHAPES>` that a shape editor
draws from. ``+ Shape`` can therefore add a body of one shape, and a front
end does not need its own answer to the question "how big is a new circle".
The sizes are bench sizes. A front end that shows kilometres is expected to
scale them to what it shows.

``snap`` carries, for each substrate, its four corners, the apex of each
face, and its middle. It also carries the points a body names for itself,
and the centre of every screw hole a body has.

The named points come before the holes. Two marks at the same place count
as one point, and the first one wins. The post hole of a mount is both a
circle in the drawing and the point the mount stands on its pedestal by,
and ``MT post`` is a more useful label than ``MT hole``.

These points come from Python because they are geometry: a corner is where
the wedge and the sagitta of a curved face put it. Beam ends are
deliberately *not* in ``snap``. The scene already carries the ends of every
beam, so a front end can offer those directly.
