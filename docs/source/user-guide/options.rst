.. _user_guide.options:

====================
Options and Settings
====================

QExPy has an options API similar to that of `pandas` to configure global behaviour related
to displaying data, methods of error propagation and more.

Options can be accessed like attributes using the "dotted-style".

.. ipython:: python

   import qexpy as q
   q.options.format.precision.sig_fig
   q.options.format.precision.sig_fig = 2
   q.options.format.precision.sig_fig

Alternatively, options can be accessed and changed using these functions:

* :func:`~qexpy.get_option` / :func:`~qexpy.set_option` - get/set the value of a single option.
* :func:`~qexpy.reset_option` - reset one or all options to their default value.

.. ipython:: python

   q.get_option('format.precision.sig_fig')
   q.set_option('format.precision.sig_fig', 2)
   q.get_option('format.precision.sig_fig')
   q.reset_option()

Available Options
~~~~~~~~~~~~~~~~~

Use :func:`~qexpy.describe_option` to see a list of available options and their descriptions.

.. ipython:: python

   q.describe_option()

You can also use it to get help on a particular option:

.. ipython:: python

   q.describe_option("format.precision.sig_fig")
