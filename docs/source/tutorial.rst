Tutorial
===============================

There are two notebooks. The first one teaches gtrace. The second one
uses gtrace on a real bench problem: matching a laser into a cavity
with two lenses.

Every cell on these pages was run from the notebook files in the
repository. The numbers and the figures you see here are the ones you
get when you run the notebooks yourself.

Running the notebooks
---------------------

Both notebooks need gtrace and nothing else from the repository.
Install gtrace with the viewer as a Jupyter widget::

    pip install "gtrace[notebook]"

This does not install Jupyter itself. If you do not have Jupyter::

    pip install jupyterlab

The first notebook needs nothing more. The mode matching notebook draws
contour maps with Matplotlib, runs one scan in parallel with joblib, and
optimises with SciPy::

    pip install matplotlib joblib scipy

Then download the notebook file from GitHub with the download button at
the top right of the page:

* `gtrace-tutorial.ipynb
  <https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/gtrace-tutorial.ipynb>`__
* `modematching.ipynb
  <https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/modematching.ipynb>`__

Open the file in JupyterLab, or in VS Code's notebook editor.

You can also clone the repository, which is what you do when you want
to change gtrace itself. See :doc:`intro`. The two notebooks are in
``docs/source/tutorial/``.

The gtrace tutorial
-------------------

This notebook builds a bench of one laser and three mirrors, and then
works on it in the viewer.

It reads the beam parameters off a beam, moves an element by dragging
it, aims an element at a beam, measures a distance across a substrate,
puts the mirrors into mounts on a breadboard, and draws a new part in
the shape editor. It then saves the layout, writes a single HTML page
you can send to somebody, and exports the drawing to DXF.

The second half explains what is underneath: the coordinates, the
beam and mirror objects, where the ghost beams come from, and the
edit messages the viewer sends. Every action in the viewer sends one
message, and you can send the same message from a notebook cell.

The last chapter builds the KAGRA input mode cleaner. This cavity is
not placed by hand. The three mirror positions and angles are computed
in plain Python first, and the result is registered in a layout
afterwards. Use this way of working when the geometry has to be
computed.

.. toctree::
   :maxdepth: 1

   tutorial/gtrace-tutorial

Worked example: mode matching a cavity
--------------------------------------

A laser has to be coupled into a Fabry-Perot cavity, using two lenses
picked from a stock of focal lengths.

The notebook computes the eigenmode of the cavity from its g
parameters, and then confirms the same answer by ray tracing. It scans
the two lens positions and draws the mode matching as a contour map with
Matplotlib, first in a plain loop and then in a parallel version that
uses joblib. The best point of the map is the starting point of a SciPy
optimisation that reaches a perfect match. The result is checked in the
viewer: on both cavity mirrors, the ROC of the beam is equal to the ROC
of the mirror.

.. toctree::
   :maxdepth: 1

   tutorial/modematching

Reference pages
---------------

:doc:`basic_concepts` and :doc:`propagation` describe the surfaces and
the matrices underneath the tutorial. :doc:`layout` is the reference
page for ``OpticalLayout``, and :doc:`viewer` for the viewer.
