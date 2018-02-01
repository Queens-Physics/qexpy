
# Check the python interpretter version
import sys
if sys.version_info[0] < 3: # No reason to assume a future Python 4 will break compatability.
    raise ImportError("Error: QExPy is only supported on Python 3. Please upgrade your interpretter.\n"
                     "If you're using Anaconda, you can download the correct version here:\n"
                     "https://www.continuum.io/downloads")


#Whether to use Bokeh or matplotlib:
plot_engine="bokeh"
plot_engine_synonyms = {"bokeh":["bokeh", "Bokeh", "Bk", "bk", "Bo", "bo", "B", "b"],  
                        "mpl":["mpl","matplotlib","MPL","Mpl","Matplotlib", "M","m"]}
quick_MC = False

#Default parameters for things:
from qexpy.defaults import settings

#Error propagation
from qexpy.error import Measurement, MeasurementArray, ExperimentalValue, \
                        set_print_style, set_sigfigs_centralvalue, set_sigfigs_error, set_sigfigs, set_error_method, show_histogram, \
                        sqrt, sin, cos, tan, sec, csc, cot, log, exp, e, asin, acos, atan        

#Plotting and fitting
from qexpy.plotting import Plot, MakePlot
from qexpy.fitting import XYDataSet, XYFitter, DataSetFromFile
from qexpy.plot_utils import bk_plot_dataset, bk_add_points_with_error_bars,\
                             bk_plot_function
    

__version__ = '1.0.3'

# The following will initialize bokeh if running in a notebook,
# and hacks the _nb_loaded variable which is required for all plots
# to show when Run All is used in a notebook. This bug arrived in
# bokeh 12.1, hopefully they get rid of it...

import qexpy.utils as qu
import bokeh.io as bi

from qexpy.utils import get_data_from_file

if qu.in_notebook():
    
    qu.mpl_output_notebook() # calls matplotlib inline
    
    bi.output_notebook()
    qu.bokeh_ouput_notebook_called = True
    '''This hack is required as there is a bug in bokeh preventing it
    from knowing that it was in fact loaded.
    '''
    bi._nb_loaded = True
