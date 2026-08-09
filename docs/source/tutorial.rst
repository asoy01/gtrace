Tutorial
===============================

Two worked notebooks. The first is the introduction to gtrace; the
second applies it to a task an optics bench keeps posing. Every cell on
these pages was executed from the files in the repository, so what you
see here is what you get when you run them.

The gtrace tutorial
-------------------

A worked introduction to gtrace. It starts from the coordinate
conventions and builds up to a traced optical system, the KAGRA input
mode cleaner, and the interactive viewer.

The first part uses the low level interface throughout: it makes
mirrors and beams and passes them around itself, drawing the results to
DXF. This is the layer described in :doc:`basic_concepts` and
:doc:`propagation`. The last part collects the same optics into an
:py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` and opens them
in the viewer, which is where clicking a beam will tell you what the
beam is doing at that point. :doc:`layout` and :doc:`viewer` describe
both in reference detail.

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

To run them rather than read them, take the source tree::

    git clone https://github.com/asoy01/gtrace.git

and open ``docs/source/tutorial/gtrace-tutorial.ipynb`` or
``docs/source/tutorial/modematching.ipynb``.
