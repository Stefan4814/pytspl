=======================================
Working with Cell Complexes in PyTSPL
=======================================

In this tutorial we introduce the new :class:`CellComplex` class, which
extends PyTSPL to handle higher-order polygons (cells), not just simplices.
You will learn how to:

1. Instantiate a cell complex from nodes, edges and polygons.
2. Visualize the cell complex.  
3. Build a cell complex from a custom dataset.
4. Convert a cell complex into a simplicial complex, and vice versa.  
5. Compute Hodge Laplacians on the cell complex.  

Prerequisites
-------------
Make sure you have installed the latest version of PyTSPL:

.. code-block:: bash

    pip install pytspl

Building a Cell Complex
------------------------
A cell complex is defined by:

  - ``nodes``: list of hashables  
  - ``edges``: list of 2-tuples  
  - ``polygons``: list of k-tuples (k≥3)  

Here we build a simple “pyramid” cell complex with a square base and a
center node:

.. code-block:: python

    from pytspl.cell_complex import CellComplex

    # define our nodes, edges and 4-gon base
    nodes    = [0, 1, 2, 3, 4]
    edges    = [(0,1),(1,2),(2,3),(3,0),   # square
                (0,4),(3,4)]   # spokes up to apex

    cc = CellComplex(
        nodes=nodes,
        edges=edges,
    )
    cc.print_summary()
    #number of nodes: 5
    #number of edges: 6
    #number of polygons: 2

Plotting Cell vs. Simplicial Complex
------------------------------------
We use the familiar :class:`~pytspl.plot_sc.SCPlot`, passing
``only_sc=False`` to plot cell complexes, or ``only_sc=True`` to view the
simplicial complexes:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> from pytspl import SCPlot
    >>> from pytspl.cell_complex import CellComplex
    >>> from pytspl.cell_complex.ccbuilder import CCBuilder
    >>>
    >>> nodes = [0, 1, 2, 3, 4]
    >>> edges = [(0,1),(1,2),(2,3),(3,0),
    ...         (0,4),(3,4)]
    >>> cc = CCBuilder(nodes=nodes, edges=edges).to_cell_complex()
    >>> # simple 2D coordinates for our pyramid
    >>> coords = {
    ...     0:(0.0,0.0), 1:(1.0,0.0),
    ...     2:(1.0,1.0), 3:(0.0,1.0),
    ...     4:(-0.5,-0.5)
    ... }
    >>> # filled cell complex
    >>> scplot1 = SCPlot(cc, coords, only_sc=False)
    >>> scplot1.draw_network(draw_orientation=True)

Loading Custom Datasets for Cell Complexes
------------------------------------------
You load your data (CSV, TNTP, incidence matrices, etc.) **exactly** as you
would for a simplicial complex. The only difference is that you need to
specify the ``only_sc`` argument as ``False`` when you call them.

.. plot::
    :context: close-figs

    >>> from pytspl.io import load_dataset
    >>> from pytspl import SCPlot
    >>>
    >>> complex, coords, _ = load_dataset(dataset="paper", only_sc=False)
    >>> complex.print_summary()
    #number of nodes: 7
    #number of edges: 10
    #number of polygons: 4

    >>> scplot = SCPlot(complex, coords, only_sc=False)
    >>> scplot.draw_network(draw_orientation=True)

Converting to a Simplicial Complex
----------------------------------
If you need a pure simplicial complex (triangles only), you can use the :meth:`to_simplicial_complex` method. 
Or if you want to convert a simplicial complex
to a cell complex, you can use the :meth:`to_cell_complex` method:

.. code-block:: python

    sc = cc.to_simplicial_complex()
    sc.print_summary()
    #number of nodes: 5
    #number of edges: 6
    #number of polygons: 1

    cc = sc.to_cell_complex()
    cc.print_summary()
    #number of nodes: 5
    #number of edges: 6
    #number of polygons: 2

