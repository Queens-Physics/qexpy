Plotting
========

This module is what will allow for both fitting and the creation of
figures. Similar to the process used in creating Measurement objects,
the creation and use of figures is based on the Plot object.
Furthermore, the actual figure itself can be rendered using both the
Bokeh and Matplotlib Python packages. While each engine has different
benifits, this document will focus on the use of the Plot object, rather
than the plotting software itself.

The Plot Object
---------------

Plots created with QExPy are stored in a variable like any other value.
This variable can then be operated on to add or change different aspects
of the plot, such as lines of fit, user-defined functions or simply the
colour of data point.

.. bokeh-plot::

    from bokeh.plotting import figure, output_file, show

    output_file("example.html")

    x = [1, 2, 3, 4, 5]
    y = [6, 7, 6, 4, 5]

    p = figure(title="example", plot_width=300, plot_height=300)
    p.line(x, y, line_width=2)
    p.circle(x, y, size=10, fill_color="white")

    show(p)
	
Using methods such as *.fit* or *.residuals* will create a best fit of
the data and display the residual output.  The *.fit* attribute also has
arguments of what type of fit is required and, if the model is not
built-in to the module, an initial guess of the fitting parameters.