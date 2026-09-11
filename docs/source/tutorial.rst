Tutorial
===============================

There are two notebooks. ``gtrace-tutorial.ipynb`` teaches gtrace.
``modematching.ipynb`` uses gtrace on a real bench problem: matching a
laser into a cavity with two lenses.

Every cell on the two tutorial pages was run from the notebook files in
the repository. The numbers and the figures you see here are the same
ones you get when you run the notebooks yourself.

Running the notebooks
---------------------

Neither notebook reads any other file from the repository, so you can
download the two ``.ipynb`` files on their own. Both notebooks need
gtrace installed, and ``modematching.ipynb`` needs three more packages.

Install gtrace with the viewer as a Jupyter widget::

    pip install "gtrace[notebook]"

This command does not install Jupyter itself. If you do not have
Jupyter::

    pip install jupyterlab

``gtrace-tutorial.ipynb`` needs nothing more. ``modematching.ipynb``
draws contour maps with Matplotlib, runs one scan in parallel with
joblib, and optimises the lens positions with SciPy::

    pip install matplotlib joblib scipy

Then download the notebook files from GitHub with the download button
at the top right of each page:

* `gtrace-tutorial.ipynb
  <https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/gtrace-tutorial.ipynb>`__
* `modematching.ipynb
  <https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/modematching.ipynb>`__

Open the notebook file in JupyterLab, or in VS Code's notebook editor.

You can also clone the repository. Clone it when you want to change
gtrace itself. :doc:`intro` gives the clone command. The two notebooks
are in ``docs/source/tutorial/``.

The gtrace tutorial
-------------------

``gtrace-tutorial.ipynb`` builds a bench of one laser and three
mirrors, and then edits the bench in the viewer.

The notebook reads the parameters of a beam, moves an element by
dragging it, aims an element at a beam, measures a distance across a
substrate, puts the mirrors into mounts on a breadboard, and draws a
new part in the shape editor. It then saves the layout, writes a single
HTML page you can send to another person, and exports the drawing to
DXF.

The second half explains how gtrace works inside: the coordinates, the
beam and mirror objects, where the ghost beams come from, and the
edit messages the viewer sends. Every action in the viewer sends one
message, and you can send the same message from a notebook cell.

The last chapter builds the KAGRA input mode cleaner. The mode cleaner
is not placed by hand. The three mirror positions and angles are
computed in plain Python first, and the result is registered in a
layout afterwards. Use the same method when the geometry has to be
computed.

.. toctree::
   :maxdepth: 1

   tutorial/gtrace-tutorial

Worked example: mode matching a cavity
--------------------------------------

A laser has to be coupled into a Fabry-Perot cavity, using two lenses
picked from a stock of focal lengths.

``modematching.ipynb`` computes the eigenmode of the cavity from its g
parameters, and then confirms the same answer by ray tracing. It scans
the two lens positions and draws the mode matching as a contour map
with Matplotlib. The scan runs first in a plain loop, then in a
parallel version that uses joblib. The best point of the map is the
starting point of a SciPy optimisation that reaches a perfect match.
The result is checked in the viewer: on both cavity mirrors, the radius
of curvature (ROC) of the beam is equal to the ROC of the mirror.

.. toctree::
   :maxdepth: 1

   tutorial/modematching

Reference pages
---------------

:doc:`basic_concepts` and :doc:`propagation` describe the surfaces and
the matrices that the tutorial uses. :doc:`layout` is the reference
page for ``OpticalLayout``, and :doc:`viewer` for the viewer.
