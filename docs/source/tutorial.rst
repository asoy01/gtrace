Tutorial
===============================

Two worked notebooks. The first is the introduction to gtrace; the
second applies it to a task an optics bench keeps posing. Every cell on
these pages was executed from the files in the repository, so what you
see here is what you get when you run them.

The gtrace tutorial
-------------------

A worked introduction to gtrace, built around the loop you actually
work in: place something, look at what the beams do, move it.

It puts three mirrors and a laser into an
:py:class:`OpticalLayout<gtrace.layout.OpticalLayout>`, opens the
result in the viewer, and adjusts it there - reading a beam off the
drawing, aiming an element by places, measuring across a substrate,
standing the optics in mounts on a breadboard and drawing a part of
its own in the shape editor. Every gesture is also a message you can
send from a cell, and the notebook shows both sides of that.

The last chapter is the KAGRA input mode cleaner, which is the other
half of the workflow: where the optics go follows from what the system
has to do, so the cavity is built and aligned in ordinary Python and
only then registered in a layout and looked at. Writing the drawing out
to DXF closes the notebook, which is where it belongs - it is how a
finished layout is handed to the rest of an engineering workflow.

:doc:`basic_concepts` and :doc:`propagation` describe the surfaces and
the matrices underneath; :doc:`layout` and :doc:`viewer` are the
reference pages for the two halves of the tutorial.

.. toctree::
   :maxdepth: 1

   tutorial/gtrace-tutorial

Worked example: mode matching a cavity
--------------------------------------

A laser is coupled into a Fabry-Perot cavity with two lenses picked
from a stock of focal lengths. The eigenmode of the cavity is computed
from its g parameters and then confirmed by ray tracing; the lens
placement is searched as a contour map of the traced mode matching
over the two positions, with a joblib-parallelised variant of the scan
as a speed example; and the best point of the map seeds an
optimisation that reaches a perfect match. The result is checked in
the viewer, where the beam ROC meets the mirror ROC on both cavity
mirrors.

.. toctree::
   :maxdepth: 1

   tutorial/modematching

Running the notebooks
---------------------

Both notebooks are self-contained: they need gtrace and nothing else
from the repository. Install it with the viewer as a Jupyter widget::

    pip install "gtrace[notebook]"

then download the notebook itself from GitHub —
`gtrace-tutorial.ipynb
<https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/gtrace-tutorial.ipynb>`__
or `modematching.ipynb
<https://github.com/asoy01/gtrace/blob/master/docs/source/tutorial/modematching.ipynb>`__,
using the download button at the top right of the GitHub page — and
open it in Jupyter or in VS Code's notebook editor.

Cloning the repository works too, and is the way to go when you mean
to change gtrace itself; see :doc:`intro`.
