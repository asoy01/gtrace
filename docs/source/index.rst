Welcome to gtrace's documentation!
==================================

gtrace traces Gaussian beams through an optical system laid out on a bench.
You place mirrors and lenses in Python and trace the beams. You look at the
result in a browser and move the elements until the layout is right. When
the layout is finished, you export the drawing to DXF.

Install it with::

    pip install "gtrace[notebook]"

Then read :doc:`intro`, which builds a bench of a laser, a lens and two
mirrors, and opens it in the viewer. :doc:`tutorial` works through a larger
example.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   basic_concepts
   propagation
   layout
   viewer
   editing
   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
