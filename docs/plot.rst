The Plot Object
===============

When plotting and fitting data, a Plot object is used. A Plot object uses either a Bokeh or matplotlib backend in order to plot data.

.. autofunction:: qexpy.plotting.MakePlot 
.. autoclass:: qexpy.plotting.Plot

Properties
----------
Plots have a few properties that you can get and set. Each property has an example of how to get and set it.

.. autoinstanceattribute:: qexpy.plotting.Plot.show_fit_results
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.bk_legend_location
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.mpl_legend_location
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.errorband_sigma
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.x_range
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.y_range
   :annotation:
.. autoinstanceattribute:: qexpy.plotting.Plot.yres_range
   :annotation:


Functions
---------

.. automethod:: qexpy.plotting.Plot.add_dataset
.. automethod:: qexpy.plotting.Plot.add_function
.. automethod:: qexpy.plotting.Plot.add_line
.. automethod:: qexpy.plotting.Plot.add_residuals
.. automethod:: qexpy.plotting.Plot.fit
.. automethod:: qexpy.plotting.Plot.show_table
.. automethod:: qexpy.plotting.Plot.set_labels
.. automethod:: qexpy.plotting.Plot.set_plot_range
.. automethod:: qexpy.plotting.Plot.show