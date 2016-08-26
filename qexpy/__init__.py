__version__ = '0.2.4'

from qexpy.error import Measurement, MeasurementArray, Measurement_Array,\
                        set_print_style, set_sigfigs_centralvalue, set_sigfigs_error, set_sigfigs, set_error_method,\
                        sqrt, sin, cos, tan, sec, csc, cot, log, exp, e, asin, acos, atan
        
from qexpy.plotting import Plot
import qexpy.utils as qu

import bokeh.plotting as bp
import bokeh.io as bi

#The following will initialize bokeh if running in a notebook,
#and hacks the _nb_loaded variable which is required for all plots
#to show when Run All is used in a notebook. This bug arrived in
#bokeh 12.1, hopefully they get rid of it...

if qu.in_notebook():
    bi.output_notebook()
    qu.bokeh_ouput_notebook_called = True
    '''This hack is required as there is a bug in bokeh preventing it
    from knowing that it was in fact loaded.
    '''
    bi._nb_loaded = True

