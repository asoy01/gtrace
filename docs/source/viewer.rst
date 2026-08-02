The viewer
===============================

Historically the only way to look at a trace was to write a DXF file and open it in CAD software. That works, but it is a slow loop, it requires a CAD program on every machine that wants to look at a result, and the DXF carries no physics: it is a drawing, so there is nothing in it to ask about the beam.

gtrace ships a viewer that answers those three points. It draws the same scene, needs no software beyond a browser, and carries the q-parameters of the beams alongside the geometry, so you can click anywhere along a beam and read out what the beam is doing *at that point* — not only at the vertices.

.. code-block:: python

    layout.show()

That is the whole entry point. In a Jupyter notebook it returns a widget that renders in the output cell; anywhere else it writes a self-contained HTML file and opens it in your browser. Both drive the same viewer.

Three ways in
--------------

The viewer is one piece of dependency-free JavaScript with three front ends over it. They share a serializer and a scene format, so they show the same picture and report the same numbers.

.. code-block:: python

    layout.show()                       # picks the right one for you
    layout.render_html('trace.html')    # a file you can send to someone
    layout.widget()                     # a notebook cell output

**Self-contained HTML** — :py:meth:`render_html<gtrace.layout.OpticalLayout.render_html>` writes one file with the scene, the viewer code and the styling all inlined. There is no server, no CDN and no install. You can mail the file to a collaborator, and it will still open in ten years.

**Notebook widget** — :py:meth:`widget<gtrace.layout.OpticalLayout.widget>` embeds the viewer as a cell output. Because the Python kernel is still alive behind it, this is the one that can edit and re-trace. It needs ``anywidget``; without it, use the HTML backend.

**Explicit choice** — ``layout.show(backend='html')`` or ``backend='widget'`` overrides the automatic pick.

If you are not using an :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>`, the renderer is callable directly, and can also be dropped into ``drawOptSys`` in place of the DXF renderer:

.. code-block:: python

    from gtrace.draw.viewer import renderHTML, html_render_func

    renderHTML(canvas, beams, 'trace.html', optics=optList)

    drawOptSys(optList, beamList, 'trace.html',
               render_func=html_render_func(beamList, optList))

Pass ``optics`` if you want to be able to click the elements. Without it the viewer draws them but has no way to say which is which.

Reading out a beam
-------------------

Click anywhere on a beam. The point is projected onto the beam segment, the q-parameter is advanced to that distance, and the panel reports, separately for the x and y directions where they differ:

============================ ==============================================
Radius ``w``                 Beam radius at the clicked point
ROC ``R``                    Radius of curvature of the wavefront there
``q``                        The complex q-parameter itself
Waist ``w₀``                 Radius at the waist of this beam
To waist                     Distance from the clicked point to the waist
``z_R``                      Rayleigh range
Gouy                         Accumulated Gouy phase
Power, wavelength, ``n``     Properties of the beam
Optical dist.                Optical path length accumulated so far
Stray order                  How many ghost reflections produced this beam
============================ ==============================================

This is why the DXF route is a dead end for interactive use, and why the viewer is not simply a DXF renderer pointed at a browser: the readout needs the beam objects, and a DXF has none.

Clicking an element instead of a beam opens its properties.

Controls
---------

Zoom with the wheel, centred on the cursor. Pan by dragging the background. Layers can be toggled individually, so the stray beams can be taken out of the way without re-running anything.

Beam widths
^^^^^^^^^^^^

The side bar chooses how the envelope is drawn: the width in units of the 1/e² radius (1 σ, 2.7 σ or 3 σ) and which transverse direction it shows (x, y or their average). The default is 2.7 σ in x. See :ref:`why-2.7-sigma` for what those numbers mean.

Changing either redraws but does not re-trace: the display changed, the physics did not. The controls are absent from the static HTML, since redrawing needs Python; choose there at write time with ``render_html(..., width_mode='y')``.

Editing
--------

In the notebook widget the loop runs both ways. Clicking an element opens a properties panel where its position, orientation, size, curvature, refractive index, reflectivities and tracing flags can be edited. Elements can be added (``+ Mirror``, ``+ CyMirror``), removed and renamed.

Each edit is applied to the registered object, the layout is re-traced, and the new scene is pushed back into the view — keeping your current zoom, pan and layer visibility, so the picture does not jump underneath you.

Since the layout holds optics by reference, the object you edited in the browser is the object your own variable names:

.. code-block:: python

    w = layout.widget()
    w                       # displays the viewer; move PRM in it
    PRM.HRcenter            # shows where you moved it to

And in the other direction:

.. code-block:: python

    PRM.HRcenter = [0.6, 0]
    w.update()              # re-traces and redraws in place

``w.edits`` returns the edit messages received so far, oldest first, which is a convenient record of what you did by hand.

Curvature is presented as a radius of curvature rather than as the ``inv_ROC_HR`` the model stores, and converted on the way in and out. A flat surface is then ``inf`` rather than a suspicious-looking zero, and the number in the panel is the number written on the mirror's data sheet.

Read-only viewers
^^^^^^^^^^^^^^^^^^

A widget constructed without a layout, or with ``editable=False``, shows the readout but no editing controls. The static HTML is always read-only: there is no Python behind it to re-trace, so an edit could not mean anything.

A rejected edit — an unknown attribute, a value outside the permitted set, a duplicate name — leaves the layout untouched and reports itself in the viewer rather than raising somewhere nothing would see it.

Saving and loading
-------------------

The side bar has a ``Layout file`` panel with a file name and Save / Load buttons. The reading and writing are done by Python, and the path is relative to where the kernel is running; the page is never given access to your disk.

Loading updates the layout in place, so the names bound in the cells above keep pointing at the right objects. See :doc:`layout` for what that means and why it matters.

Saving changes nothing on screen, so the viewer says so in the status line — otherwise there would be no way to tell whether the button did anything.

DXF is still there
-------------------

None of this replaces the DXF output. ``renderDXF`` is unchanged, and a layout drawn to a canvas can be rendered to DXF exactly as before:

.. code-block:: python

    import gtrace.draw.renderer as renderer

    renderer.renderDXF(layout.draw(), 'layout.dxf')

The viewer is for looking and adjusting; DXF remains the way to hand a layout to the rest of an engineering workflow.
