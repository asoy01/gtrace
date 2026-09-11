Welcome to gtrace's documentation!
==================================

gtrace traces Gaussian beams through an optical system laid out on a bench.
You place mirrors and lenses in Python and trace the beams. You look at the
result in gtrace's viewer and move the elements until the layout is right.
The viewer runs as a widget inside a Jupyter notebook, or as a standalone
HTML page in a web browser. When the layout is finished, you export the
drawing to DXF.

Install gtrace with::

    pip install "gtrace[notebook]"

This command does not install Jupyter itself. If you do not have Jupyter::

    pip install jupyterlab

Then read :doc:`intro`. That page builds a bench of a laser, a lens and two
mirrors, and opens the bench in the viewer. :doc:`tutorial` works through a
larger example.

The other pages are reference:

* :doc:`basic_concepts`: the objects a layout is made of.
* :doc:`propagation`: the three ways to move a beam.
* :doc:`layout`: the ``OpticalLayout`` object, which holds a whole system.
* :doc:`viewer`: the viewer and its controls.
* :doc:`editing`: the messages the viewer sends to change a layout.

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
