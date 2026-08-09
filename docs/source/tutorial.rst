Tutorial
===============================

A worked introduction to gtrace, as a Jupyter notebook. It starts from
the coordinate conventions and builds up to a traced optical system, the
KAGRA input mode cleaner, and the interactive viewer.

.. toctree::
   :maxdepth: 1

   tutorial/gtrace-tutorial
   tutorial/modematching

To run it rather than read it, take the source tree::

    git clone https://github.com/asoy01/gtrace.git

and open ``docs/source/tutorial/gtrace-tutorial.ipynb``. Every cell on
the page above was executed from that file, so what you see here is what
you get when you run it.

What it covers
---------------

The first part uses the low level interface throughout: it makes mirrors
and beams and passes them around itself, drawing the results to DXF.
This is the layer described in :doc:`basic_concepts` and
:doc:`propagation`.

The last part collects the same optics into an
:py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` and opens them in
the viewer, which is where clicking a beam will tell you what the beam
is doing at that point. :doc:`layout` and :doc:`viewer` describe both in
reference detail.
