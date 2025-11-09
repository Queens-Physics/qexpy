.. _user_guide.options:

====================
Options and Settings
====================

QExPy has an options API similar to that of `pandas` to configure global 
behaviour of data display, methods of error propagation and more.

Options can be accessed like attributes using the "dotted-style".

.. ipython:: python

   import qexpy as q
   q.options.format.precision.sigfigs
   q.options.format.precision.sigfigs = 2
   q.options.format.precision.sigfigs

Alternatively, options can be accessed and changed using these functions:

* :func:`~qexpy.get_option` / :func:`~qexpy.set_option` - retrieve/modify the value of an option.
* :func:`~qexpy.reset_options` - reset one or all options to their default value(s).

.. ipython:: python

   q.get_option('format.precision.sigfigs')
   q.set_option('format.precision.sigfigs', 2)
   q.get_option('format.precision.sigfigs')
   q.reset_options()

Available Options
~~~~~~~~~~~~~~~~~

Use :func:`~qexpy.describe_option` to see a list of available options and their descriptions.

.. ipython:: python

   q.describe_option()

You can also use it to get help on a particular option:

.. ipython:: python

   q.describe_option("format.precision.sigfigs")
