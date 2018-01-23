import numpy as np
import qexpy as q
import qexpy.error as qe
import qexpy.utils as qu
import qexpy.fitting as qf
import qexpy.plot_utils as qpu

import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.layouts as bl
import bokeh.models as mo

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

from ipywidgets import interact 

CONSTANT = qu.number_types
ARRAY = qu.array_types



def MakePlot(xdata=None, ydata=None, xerr=None, yerr=None, data_name=None,
             dataset=None, xname=None, xunits=None, yname=None, yunits=None):
    '''Use this function to create a plot object, by providing either arrays
    corresponding to the x and y data, Measurement_Arrays for x and y, or
    an XYDataSet. If providing a dataset, it can be specified as either the 
    x argument or the dataset argument.

    :param xdata: The data to be plotted on the x axis.
    :type xdata: array, Measurement_Array, XYDataSet
    :param ydata: The data to be plotted on the y axis.
    :type ydata: array, Measurement_Array
    :param xerr: The error on the data on the x axis.
    :type xerr: array
    :param yerr: The error on the data on the y axis.
    :type yerr: array
    :param data_name: The name of the data to be plotted.
    :type data_name: str
    :param dataset: An XYDataSet to be plotted.
    :type dataset: XYDataSet
    :param xname: The name x axis.
    :type xname: str
    :param xunits: The units of the x axis.
    :type xname: str
    :param yname: The name y axis.
    :type xname: str
    :param yunits: The units of the y axis.
    :type yunits: str
    '''
    
    if xdata is None and ydata is None and dataset is None:
        return Plot(None)
    
    elif xdata is not None and ydata is None:
        #assume that x is a dataset:
        if isinstance(xdata, qf.XYDataSet):
            if xname is not None and isinstance(xname, str):
                xdata.xname = xname
            if yname is not None and isinstance(yname, str):
                xdata.yname = yname           
            if xunits is not None and isinstance(xunits, str):
                xdata.xunits = xunits
            if yunits is not None and isinstance(yunits, str):
                xdata.yunits = yunits 
            if data_name is not None and isinstance(data_name, str):
                xdata.name = data_name
            return Plot(xdata)
        else:
            print("Must specify x AND y or dataset, returning empty plot")
            return Plot(None)
        
    elif dataset is not None:
        if xname is not None and isinstance(xname, str):
            dataset.xname = xname
        if yname is not None and isinstance(yname, str):
            dataset.yname = yname           
        if xunits is not None and isinstance(xunits, str):
            dataset.xunits = xunits
        if yunits is not None and isinstance(yunits, str):
            dataset.yunits = yunits 
        if data_name is not None and isinstance(data_name, str):
            dataset.name = data_name
        return Plot(dataset)
  
    elif (xdata is not None and ydata is not None):
        ds = qf.XYDataSet(xdata, ydata, xerr=xerr, yerr=yerr, data_name=data_name,
                          xname=xname, xunits=xunits, yname=yname, yunits=yunits)
        return Plot(dataset = ds)
    
    else:
        return Plot(None)

class Plot:
    '''Object for plotting and fitting datasets built on
    Measurement_Arrays

    The Plot object holds a list of XYDataSets, themselves containing
    pairs of MeasurementArrays holding x and y values to be plotted and 
    fitted. The Plot object uses bokeh or matplotlib to display the data,
    along with fit functions, and user-specified functions. One should configure
    the various aspects of the plot, and then call the show() function
    which will actually build the plot object and display it. 

    :param dataset: The dataset to be plotted.
    :type dataset: XYDataSet
    '''

    def __init__(self, dataset=None):
        '''
        Constructor to make a plot based on a dataset.

        :param dataset: The dataset to be plotted.
        :type dataset: XYDataSet
        '''                                    
    
        #Colors to be used for coloring elements automatically
        self.color_palette = q.settings["plot_color_palette"]
        self.color_count = 0

        #Dimensions of the figure in pixels
        self.dimensions_px = [ q.settings["plot_fig_x_px"], 
                               q.settings["plot_fig_y_px"] ]
        #Screen dots per inch, required for mpl
        self.screen_dpi = q.settings["plot_screen_dpi"]
                
        #Where to save the plot              
        self.save_filename = 'myplot.html'
        '''A string of what the name of the plot will be when saved.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.fit('linear')       # Will save file as 'linear_fit.html'
            fig.show()
        ''' 
        
        #How big to draw error bands on fitted functions
        self.errorband_sigma = 1
        '''An integer of the number of sigmas to show in the error band.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.errorband_sigma = 2  # Will draw 2 sigmas on the fit
            fig.fit('linear')

            fig.show()
        '''
        #whether to show residuals
        self.show_residuals=False
        #whether to include text labels on plot with fit parameters
        self.show_fit_results=True 
        '''A boolean of whether to show the results from fitting the data.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6, 7, 8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.show_fit_results = False    # Will prevent the linear fit from printing results

            fig.show()
        '''
        self.fit_results_x_offset=0
        self.fit_results_y_offset=0
        
        #location of legend
        self.bk_legend_location = "top_left"
        '''A string of the location of the legend when using Bokeh to plot.
        Acceptable values can be found here: http://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#inside-the-plot-area

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6, 7, 8], error=0.1)
            q.plot_engine = 'bokeh'

            fig = q.MakePlot(x, y)
            fig.bk_legend_location = 'top_right'    # Moves the legend to the top right
            fig.show()
        '''
        self.bk_legend_orientation = "vertical"
        self.mpl_legend_location = "upper left"
        '''A string of the location of the legend when using MatPlotLib to plot.
        Acceptable values can be found here: https://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6, 7, 8], error=0.1)
            q.plot_engine = 'mpl'

            fig = q.MakePlot(x, y)
            fig.mpl_legend_location = 'upper right'    # Moves the legend to the top right
            fig.show()
        '''
        self.mpl_show_legend = True
        
        #The data to be plotted are held in a list of datasets:
        self.datasets=[]
        #Each data set has a color, so that the user can choose specific
        #colors for each dataset
        self.datasets_colors=[]      
        
        #Functions to be plotted are held in a list of functions
        self.user_functions = []
        self.user_functions_count=0
        self.user_functions_pars = []
        self.user_functions_names = []
        self.user_functions_colors = []
        self.user_functions_sigmas = []
        
        #Add margins to the range of the plot
        self.x_range_margin = 0.5
        self.y_range_margin = 0.5 
        #Default range for the axes
        self.x_range = [0,1]
        '''A list of the minimum x value and the maximum x value.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6, 7, 8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.x_range = [0, 5]    # Will plot from x=0 to x=5
            fig.show()
        '''
        self.y_range = [0,1]
        '''A list of the minimum y value and the maximum y value.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6, 7, 8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.y_range = [4, 9]    # Will plot from y=4 to y=9
            fig.show()
        '''
        self.yres_range = [0,0.1]
        '''A list of the minimum y value and the maximum y value for the residuals.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            fig = q.MakePlot(x, y)
            fig.fit('linear')
            fig.show_residuals = True
            fig.yres_range = [-0.5, 0.5]    # Will plot residuals from y=-0.5 to y=0.5
            fig.show()
        '''
        
        #Labels for axes
        self.axes = {'xscale': 'linear', 'yscale': 'linear'}         
        self.labels = {
            'title': "y as a function of x",
            'xtitle': "x",
            'ytitle': "y"}

        self.lines = {'x':[],
                      'y':[]}
        
        if dataset != None:
            self.datasets.append(dataset)            
            self.datasets_colors.append(self._get_color_from_palette())
            self.initialize_from_dataset(dataset)
        else:
            self.initialized_from_dataset = False

    def show_table(self, latex=False):
        '''Prints the data of the Plot in a formatted table.

        :param latex: Whether to print the data using Latex formatting.
        :type show: bool

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)
            figure.show_table()
        '''
        dataset = self.datasets[-1]
        dataset.show_table(latex=latex)

    def initialize_from_dataset(self, dataset):
        '''Initialize axes labels and ranges from the dataset'''
        self.labels = {
            'title': dataset.name,
            'xtitle': dataset.xname +' ['+dataset.xunits+']',
            'ytitle': dataset.yname +' ['+dataset.yunits+']'}
        
        #Get the range from the dataset (will include the margin)
        self.set_range_from_dataset(dataset)
        self.initialized_from_dataset = True
        
    def _get_color_from_palette(self):
        '''Automatically select a color from the palette and increment
        the color counter'''
        self.color_count += 1
        if self.color_count>len(self.color_palette):
            self.color_count = 1
        return self.color_palette[self.color_count-1]
    
    def check_datasets_color_array(self):
        '''Make sure that color array is the same length as dataset array'''
        if len(self.datasets) == len(self.datasets_colors):
            return
        elif len(self.datasets) > len(self.datasets_colors):
            for i in range(len(self.datasets_colors), len(self.datasets)):
                self.datasets_colors.append(self._get_color_from_palette())
        elif len(self.datasets_colors) > len(self.datasets):
            while len(self.datasets_colors) != len(self.datasets):
                self.datasets_colors.pop()
        else: pass
        
    def check_user_functions_color_array(self):
        '''Make sure that color array is the same length as function array'''
        if len(self.user_functions) == len(self.user_functions_colors):
            return
        elif len(self.user_functions) > len(self.user_functions_colors):
            for i in range(len(self.user_functions_colors), len(self.user_functions)):
                self.user_functions_colors.append(self._get_color_from_palette())
        elif len(self.user_functions_colors) > len(self.user_functions):
            while len(self.user_functions_colors) != len(self.user_functions):
                self.user_functions_colors.pop()
        else: pass                     
    
    def set_range_from_datasets(self):
        '''Make sure the x and y range can accomodate all datasets. 
        Expands the range if needed.
        '''
        for ds in self.datasets:
            xr = ds.get_x_range(self.x_range_margin)
            yr = ds.get_y_range(self.y_range_margin)
            self.x_range = [min(xr[0], self.x_range[0]), max(xr[1], self.x_range[1])]
            self.y_range = [min(yr[0], self.y_range[0]), max(yr[1], self.y_range[1])]
        
    def set_range_from_dataset(self, dataset):
        '''Use a dataset to set the range for the figure. 
        Adds margins of 5%. of the range.
        '''
        xr = dataset.get_x_range()
        self.x_range_margin = (xr[1]-xr[0])*0.05
        xr_margin = dataset.get_x_range(self.x_range_margin)

        yr = dataset.get_y_range()
        self.y_range_margin = (yr[1]-yr[0])*0.05
        yr_margin = dataset.get_y_range(self.y_range_margin)

        self.x_range = xr_margin
        self.y_range = yr_margin
        
    def set_yres_range_from_fits(self):
        '''Set the range for the residual plot, based on all datasets that
        have a fit.'''      
        for dataset in self.datasets:
            if dataset.nfits > 0:
                yr = dataset.get_yres_range(self.y_range_margin)
                self.yres_range = [min(yr[0], self.yres_range[0]), max(yr[1], self.yres_range[1])]               
        
    def fit(self, model=None, parguess=None, fit_range=None, print_results=None,
            datasetindex=-1, fitcolor=None, name=None, sigmas=None):
        '''Fit a dataset to model - calls XYDataSet.fit and returns a 
        Measurement_Array of fitted parameters

        :param model: The fit function
        :type model: Function
        :param parguess: Guesses for the parameters of the fit.
        :type parguess: list
        :param fit_range: The range over which the fit function is to be shown.
        :type fit_range: list
        :param print_results: Whether to display the results of the fit.
        :type print_results: bool
        :param datasetindex: The index of the XYDataSet to be fit.
        :type datasetindex: int
        :param fitcolor: The color of the fit.
        :type fitcolor: str
        :param name: The name of the fit.
        :type name: str
        :param sigmas: How many sigmas to include in the error band.
        :type sigmas: int

        :returns: The parameters of the fit
        :rtype: Measurement_Array

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)
            figure.fit('linear')

            figure.show()
        '''
        if sigmas == None:
            sigmas = self.errorband_sigma

        if print_results == None:
            print_results = self.show_fit_results
        results = self.datasets[datasetindex].fit(model, parguess, fit_range, fitcolor=fitcolor, name=name, print_results=print_results, sigmas=sigmas) 
        return results
        
    def print_fit_parameters(self, dataset=-1):
        if len(self.datasets)<1:
            print("No datasets")
            return
        if self.datasets[-1].nfits>0:
            print("Fit parameters:\n"+str(self.datasets[dataset].fit_pars[-1]))    
        else:
            print("Datasets have not been fit")
            
    def get_dataset(self, index=-1):
        if len(self.datasets) > 0:
            if index < len(self.datasets) -1:
                return self.datasets[index]
            else:
                return None
        else:
            return None
        
###############################################################################
# User Methods for adding to Plot Objects
###############################################################################

    def add_residuals(self):
        '''Add a subfigure with residuals to the main figure when plotting.

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)
            figure.fit('linear')
            figure.add_residuals()

            figure.show()
        '''
        self.set_yres_range_from_fits()
        for ds in self.datasets:
            if ds.nfits > 0:
                self.show_residuals = True
                return

    
    def add_function(self, function, pars = None, name=None, color=None, x_range=None, sigmas=None):
        '''Add a user-specifed function to the list of functions to be plotted.
        
        All datasets are functions when populate_bokeh_figure is called
        - usually when show() is called.

        :param function: The function to be added to the Plot.
        :type function: Function
        :param pars: The parameters of the function.
        :type pars: list, array, Measurement_Array
        :param name: The name of the function.
        :type name: str
        :param color: The color of the function.
        :type color: str
        :param x_range: The range on which the function is to be plotted.
        :type x_range: array
        :param sigmas: How many sigmas to include in the error band.
        :type sigmas: int

        .. code-block:: python

            def func(x, *pars):
                return pars[0] + pars[1]*x

            figure = q.MakePlot()

            # This function is not related to any data.
            figure.add_function(func, name="Function", pars = [1, 5],
               color = 'saddlebrown', x_range =[-10,10])

            figure.show()
        '''

        if sigmas == None:
            sigmas = self.errorband_sigma
        
        if x_range is not None:
            if not isinstance(x_range, ARRAY):
                print("Error: x_range must be specified as an array of length 2")
            elif len(x_range) != 2:
                print("Error: x_range must be specified as an array of length 2")
            else:
                self.x_range[0]=x_range[0]-self.x_range_margin
                self.x_range[1]=x_range[1]+self.x_range_margin
                
        xvals = np.linspace(self.x_range[0],self.x_range[1], 100)
        
        #check if we should change the y-axis range to accomodate the function
        fvals = np.zeros(xvals.size)
        ferr = fvals
        if not isinstance(pars, np.ndarray) and pars == None:
            fvals = function(xvals)
        elif isinstance(pars, qe.Measurement_Array) :
            recall = qe.Measurement.minmax_n
            qe.Measurement.minmax_n=1
            fmes = function(xvals, *(pars))
            fvals = fmes.means
            ferr = fmes.stds
            qe.Measurement.minmax_n=recall
        elif isinstance(pars,(list, np.ndarray)):
            fvals = function(xvals, *pars)
        else:
            print("Error: Not a recognized format for parameter")
            return
                 
        fmax = (fvals+ferr).max() + self.y_range_margin
        fmin = (fvals-ferr).min() - self.y_range_margin
        
        if fmax > self.y_range[1]:
            self.y_range[1]=fmax
        if fmin < self.y_range[0]:
            self.y_range[0]=fmin
            
        self.user_functions.append(function)
        self.user_functions_pars.append(pars)
        self.user_functions_sigmas.append(sigmas)
        fname = "userf_{}".format(self.user_functions_count) if name==None else name
        self.user_functions_names.append(fname)
        self.user_functions_count +=1
        
        if color is None:
            self.user_functions_colors.append(self._get_color_from_palette())
        else: 
            self.user_functions_colors.append(color)
        
    def add_dataset(self, dataset, color=None, name=None):
        '''Add a dataset to the Plot object. All datasets are plotted
        when populate_bokeh_figure is called - usually when show() is called

        :param dataset: The dataset to be added to the plot.
        :type dataset: XYDataSet
        :param color: The color of the dataset.
        :type color: str
        :param name: The name of the dataset.
        :type name: str

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)
            xy = q.XYDataSet(x, y)

            fig = q.MakePlot()
            fig.add_dataset(xy)
            fig.show()
        '''
        self.datasets.append(dataset)
        
        if len(self.datasets) < 2:    
            self.initialize_from_dataset(self.datasets[0])
        else:
            self.set_range_from_dataset(dataset)
            
        if color is None:
            self.datasets_colors.append(self._get_color_from_palette())
        else: 
            self.datasets_colors.append(color)
        if name != None:
            self.datasets[-1].name=name
          
        self.set_range_from_datasets()
            
        self.set_yres_range_from_fits()

    def add_line(self, x=None, y=None, color='black', dashed=False):
        '''Add a vertical or horizontal line to the Plot object.

        :param x: The x position of the line to be drawn 
                  (x or y position can be specified, but not both).
        :type x: float
        :param y: The y position of the line to be drawn
                  (x or y position can be specified, but not both).
        :type y: float
        :param color: The color of the line to be drawn.
        :type color: str
        :param dashed: Whether to draw a dashed line.
        :type dashed: bool

        .. code-block:: python

            figure = q.MakePlot()

            # Adds a vertical line at x = 10
            figure.add_line(x=10)

            figure.show()
        '''
        x_pos = x
        y_pos = y
        if bool(x_pos) == bool(y_pos): # Can't have both x and y data
            print('''Lines must be given either an x or a y value, but not both.''')
        if type(x_pos) in CONSTANT:
            x_line_data = {'pos':float(x_pos), 'color':color, 'dashed':dashed}
            self.lines['x'].append(x_line_data)
        elif type(y_pos) in CONSTANT:
            y_line_data = {'pos':float(y_pos), 'color':color, 'dashed':dashed}
            self.lines['y'].append(y_line_data)
        else:
            print('''Line input must be a number.''')

#
###############################################################################
# Methods for changing parameters of Plot Object
###############################################################################
        
    def set_plot_range(self, x_range=None, y_range=None):
        '''Set the range for the figure.

        :param x_range: The x range of the graph.
        :type x_range: array
        :param y_range: The y range of the graph.
        :type y_range: array

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)
            figure.set_plot_range(x_range=[0, 5], y_range=[4, 9])

            figure.show()
        '''
        if type(x_range) in ARRAY and len(x_range) is 2:
            self.x_range = x_range
        elif x_range is not None:
            print('''X range must be a list containing a minimum and maximum
            value for the range of the plot.''')

        if type(y_range) in ARRAY and len(y_range) is 2:
            self.y_range = y_range
        elif y_range is not None:
            print('''Y range must be a list containing a minimum and maximum
            value for the range of the plot.''')

            
    def set_labels(self, title=None, xtitle=None, ytitle=None):
        '''Change the labels for plot axis, datasets, or the plot itself.

        Method simply overwrites the automatically generated names used in
        the Bokeh plot.

        :param title: The title of the plot.
        :type title: str
        :param xtitle: The title of the x axis.
        :type xtitle: str
        :param ytitle: The title of the y axis.
        :type ytitle: str

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)
            figure.set_labels("x vs y", "x_data", "y_data")

            figure.show()
        '''
        if title is not None:
            self.labels['title'] = title

        if xtitle is not None:
            self.labels['xtitle'] = xtitle

        if ytitle is not None:
            self.labels['ytitle'] = ytitle


    def resize_plot_px(self, width=None, height=None):
        
        if width is None:
            width = 600
        if height is None:
            height = 400
        self.dimensions_px = [width, height]


    def show(self, output='inline', populate_figure=True, refresh=True):
        '''
        Show the figure, will call one of the populate methods
        by default to build a figure.

        :param output: How the histogram is to be output. Can be 'inline' or 'file'.
        :type output: str
        :param populate_figure: Whether the figure needs to be populated.
        :type populate_figure: bool
        :param refresh: Whether the mpl figure should be refreshed.
        :type refresh: bool

        .. code-block:: python

            x = q.MeasurementArray([1, 2, 3, 4], error=0.5)
            y = q.MeasurementArray([5, 6.2, 7, 7.8], error=0.1)

            figure = q.MakePlot(x, y)

            figure.show()
        '''
        if q.plot_engine in q.plot_engine_synonyms["bokeh"]:
            self.set_bokeh_output(output)      
            if populate_figure:
                bp.show(self.populate_bokeh_figure(), notebook_handle=True)
            else:
                bp.show(self.bkfigure, notebook_handle=True)
                
        elif q.plot_engine in q.plot_engine_synonyms["mpl"]:
            self.set_mpl_output(output)
            if populate_figure:
                self.populate_mpl_figure(refresh=refresh)
            if output == 'file':
                plt.savefig(self.user_save_filename, bbox_inches='tight')
            plt.show()
            
        else:
            print("Error: unrecognized plot engine")

###############################################################################
# Methods for Returning or Rendering Matplotlib
###############################################################################  

    def set_mpl_output(self, output='inline'):
        '''Choose where to output (in a notebook or to a file)'''
        #TODO not tested, the output notebook part does not work
        
        if output == 'file' or not qu.in_notebook():
            #Prompt user for filename
            self.user_save_filename = input('Enter a filename: ')
            #TODO Decide what to do about this
            #plt.savefig(self.save_filename, bbox_inches='tight')
        elif not qu.mpl_ouput_notebook_called:
            qu.mpl_output_notebook()
            # This must be the first time calling output_notebook,
            # keep track that it's been called:
            qu.mpl_ouput_notebook_called = True
        else:
            pass
        
        
    def populate_mpl_figure(self, refresh = True):
        '''Thia is the main function to populate the matplotlib figure. It will create
        the figure, and then draw all of the data sets, their residuals, their fits,
        and any user-supplied functions'''
        
       
        if not hasattr(self, 'mplfigure_main_ax') or refresh == True:
            self.initialize_mpl_figure()
                   
        #Plot the data sets
        self.check_datasets_color_array()
        for dataset, color in zip(self.datasets, self.datasets_colors):           
            self.mpl_plot_dataset(dataset, color, show_fit_function=True,
                                  show_residuals=self.show_residuals)
            
        #Add a box with results from the fits                        
        if self.show_fit_results:
            self.mpl_show_fit_results_box(self.datasets)
            
            
        #Now add any user defined functions:
        #The range over which to plot the functions:
        xvals = [self.x_range[0]+self.x_range_margin, 
                 self.x_range[1]-self.x_range_margin]
        self.check_user_functions_color_array()
        for func, pars, fname, color, sigmas in zip(self.user_functions,
                                                    self.user_functions_pars, 
                                                    self.user_functions_names,
                                                    self.user_functions_colors,
                                                    self.user_functions_sigmas):
        
            self.mpl_plot_function(function=func, xdata=xvals,pars=pars, n=q.settings["plot_fcn_npoints"],
                               legend_name= fname, color=color, sigmas=sigmas)
        # Adds lines to the plot
        if self.lines['x'] or self.lines['y']:
            self.mpl_add_lines(self.lines)

        if self.mpl_show_legend:
            self.mplfigure_main_ax.legend(loc=self.mpl_legend_location,
                                          fontsize = q.settings["plot_fig_leg_ftsize"])

    def initialize_mpl_figure(self):
        '''Build a matplotlib figure with the desired size to draw on'''
                
        #Create the main figure object
        self.mplfigure = plt.figure(figsize=(self.dimensions_px[0]/self.screen_dpi,
                                             self.dimensions_px[1]/self.screen_dpi))
        
        #If we're showing residuals, create the axes differently,
        #create axes for the residuals, and label them
        if self.show_residuals:
            self.mplfigure_main_ax = self.mplfigure.add_axes([0,0.35,1.,0.65])
            self.mplfigure_res_ax = self.mplfigure.add_axes([0,0,1.0,0.3],
                                                            sharex=self.mplfigure_main_ax)    
            self.set_yres_range_from_fits()
            self.mplfigure_res_ax.set_ylim([self.yres_range[0], self.yres_range[1]])
            self.mplfigure_res_ax.set_xlabel(self.labels['xtitle'],
                                             fontsize=q.settings["plot_fig_xtitle_ftsize"])
            self.mplfigure_res_ax.set_ylabel("Residuals",
                                             fontsize=q.settings["plot_fig_ytitle_ftsize"])
            self.mplfigure_res_ax.grid()
            
        else:
            #TODO The latter fixes the issues of axis not showing when
            #PyPlot is shown outside of notebook. Need to translate to
            # residual case and decide if this is wanted.

            #self.mplfigure_main_ax = self.mplfigure.add_axes([0,0,1.,1.])
            self.mplfigure_main_ax = self.mplfigure.add_subplot(111)
        
        #Regardless of residuals, create the main axes
        self.mplfigure_main_ax.axis([self.x_range[0], self.x_range[1], 
                 self.y_range[0], self.y_range[1]])
        self.mplfigure_main_ax.set_xscale(self.axes['xscale'])
        self.mplfigure_main_ax.set_yscale(self.axes['yscale'])
        
        self.mplfigure_main_ax.set_xlabel(self.labels['xtitle'],
                                          fontsize=q.settings["plot_fig_xtitle_ftsize"])
        self.mplfigure_main_ax.set_ylabel(self.labels['ytitle'],
                                          fontsize=q.settings["plot_fig_ytitle_ftsize"])
        self.mplfigure_main_ax.set_title(self.labels['title'],
                                         fontsize=q.settings["plot_fig_title_ftsize"])
        self.mplfigure_main_ax.grid()  

        return self.mplfigure

    def mpl_add_lines(self, lines):
        '''Adds vertical and horizontal lines to an mpl plot.'''
        if not hasattr(self, 'mplfigure_main_ax'):
            self.initialize_mpl_figure()

        for x_data in lines['x']:
            dashed = 'dashed' if x_data['dashed'] else 'solid'
            self.mpl_plot([x_data['pos']]*2, self.y_range, color=x_data['color'], ls=dashed, lw=1, zorder=5)

        for y_data in lines['y']:
            dashed = 'dashed' if y_data['dashed'] else 'solid'
            self.mpl_plot(self.x_range, [y_data['pos']]*2, color=y_data['color'], ls=dashed, lw=1, zorder=5)
                       
    def mpl_show_fit_results_box(self, datasets=None, add_space = True):
        '''Show a box with the fit results for the given list of datasets. If 
        datasets==None, it will use self.datasets'''
        
        if not hasattr(self, 'mplfigure_main_ax'):
            self.initialize_mpl_figure()
        
        if datasets == None:
            datasets = self.dataset
            
        #Add some space to plot for the fit results:
        if add_space:
            newy = self.y_range[1]
            pix2y = newy / self.mplfigure_main_ax.patch.get_window_extent().height
            #TODO textheight should depend on font size
            textheight = 25
            pixelcount = 0
            for dataset in datasets:
                if dataset.nfits > 0:
                    pixelcount += dataset.fit_npars[-1] * textheight
            newy += pixelcount * pix2y
            self.mplfigure_main_ax.axis([self.x_range[0], self.x_range[1], 
                                        self.y_range[0], newy])
            
            
        textfit = ""
        for ds in datasets:
            if ds.nfits == 0:
                continue
            for i in range(ds.fit_npars[-1]):
                short_name =  ds.fit_pars[-1][i].__str__().split('_')
                textfit += short_name[0]+"_"+short_name[-1]+"\n"
            textfit += "\n"
            
        start_x = 0.99 + self.fit_results_x_offset
        start_y = 0.99 + self.fit_results_y_offset
        
        an = self.mplfigure_main_ax.annotate(textfit,xy=(start_x, start_y),
                                             fontsize=q.settings["plot_fig_fitres_ftsize"],
                                             horizontalalignment='right',verticalalignment='top',
                                             xycoords = 'axes fraction',
                                             bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
        return an
                    
        
    def mpl_plot_dataset(self, dataset, color='black', show_fit_function=True, show_residuals=True, fit_index = -1):
        '''Add a dataset, its fit function and its residuals to the main figure.
        It is better to use add_function() and to let populate_mpl_plot() actually
        add the function.
        '''
        
        index = fit_index if fit_index < dataset.nfits else -1
        
        if not hasattr(self, 'mplfigure_main_ax'):       
            if show_residuals:
                if dataset.nfits > 0:
                    self.show_residuals = True
            self.initialize_mpl_figure()
        
            
        if dataset.is_histogram:
            if hasattr(dataset, 'hist_data'):
                self.mplfigure_main_ax.hist(dataset.hist_data, bins=dataset.hist_bins,
                    label=dataset.name, color=color, alpha=0.7)
            else:
                self.mplfigure_main_ax.bar(dataset.xdata, dataset.ydata, width = dataset.xdata[-1]-dataset.xdata[-2],
                      label=dataset.name, color=color, alpha=0.7)
            
        else:   
            self.mplfigure_main_ax.errorbar(dataset.xdata, dataset.ydata,
                                            xerr=dataset.xerr,yerr=dataset.yerr,
                                            fmt='o',color=color,markeredgecolor = 'none',
                                             label=dataset.name)
        if dataset.nfits > 0 and show_fit_function:
            self.mpl_plot_function(function=dataset.fit_function[index],
                                   xdata=dataset.xdata,
                                   pars=dataset.fit_pars[index], n=q.settings["plot_fcn_npoints"],
                                   legend_name=dataset.fit_function_name[index],
                                   color=color if dataset.fit_color[index] is None else dataset.fit_color[index],
                                   sigmas=dataset.xyfitter[index].sigmas)
            
            if self.show_residuals and hasattr(self, 'mplfigure_res_ax') and show_residuals:           
                self.mplfigure_res_ax.errorbar(dataset.xdata, dataset.fit_yres[index].means,
                                               xerr=dataset.xerr,yerr=dataset.yerr,
                                               fmt='o',color=color,markeredgecolor = 'none')
            
                    
     
    def mpl_plot_function(self, function, xdata, pars=None, n=100,
                      legend_name=None, color='black', sigmas=None):
        '''Add a function to the main figure. It is better to use add_function() and to
        let populate_mpl_plot() actually add the function.
        
        The function can be either f(x) or f(x, *pars), in which case, if *pars is
        a Measurement_Array, then error bands will be drawn
        '''
        if sigmas == None:
            sigmas = self.errorband_sigma

        if not hasattr(self, 'mplfigure_main_ax'):
            self.initialize_mpl_figure()
      
        xvals = np.linspace(min(xdata), max(xdata), n)
        
        if pars is None:
            fvals = function(xvals)
        elif isinstance(pars, qe.Measurement_Array):
            recall = qe.Measurement.minmax_n
            qe.Measurement.minmax_n=1
            fmes = function(xvals, *pars)
            fvals = fmes.means
            ferr = fmes.stds
            qe.Measurement.minmax_n=recall
        elif isinstance(pars,(list, np.ndarray)):
            fvals = function(xvals, *pars)
        else:
            print("Error: unrecognized parameters for function")
            pass
        
        self.mplfigure_main_ax.plot(xvals,fvals, color=color, label = legend_name, zorder=5)
        
        if isinstance(pars, qe.Measurement_Array):
            for i in range(1, int(sigmas)+1):
                fmax = fvals + i*ferr
                fmin = fvals - i*ferr
                self.mplfigure_main_ax.fill_between(xvals, fmin, fmax, facecolor=color,
                                                    alpha=0.3/(sigmas-0.3*(i-1)), edgecolor = 'none',
                                                    interpolate=True, zorder=0)
        
    def interactive_linear_fit(self, error_range=5, randomize = False,
                               show_chi2 = True, show_errors=True,
                               x_range=None, y_range=None):
        '''Fits the last dataset to a linear function and displays the
        result as an interactive fit
        '''
                
        if not qu.in_notebook():
            print("Can only use this feature in a notebook, sorry")
            return
        
        if len(self.datasets) >1:
            print("Warning: only using the last added dataset, and clearing previous fits")
                     
        dataset = self.datasets[-1]
        color = self.datasets_colors[-1]
        
        dataset.clear_fits()
        dataset.fit("linear")
        
        func = dataset.fit_function[-1]
        pars = dataset.fit_pars[-1]
        parmeans = pars.means
        fname = "linear"     
        
        #Reset the range
        self.x_range = [0,1]
        self.y_range = [0,1]
        self.set_range_from_dataset(dataset)
        
        #Extend the x range to 0 if needed
        if self.x_range[0] > -0.5:
            self.x_range[0] = -0.5
            
        #make sure the y range is large enough
        xvals = np.array([self.x_range])
        fvals = dataset.fit_function[-1](xvals, *parmeans)
        fmaxvals = fvals+error_range*max(dataset.yerr)
        fminvals = fvals-error_range*max(dataset.yerr)
        self.y_range[0] = fminvals.min()
        self.y_range[1] = fmaxvals.max()

        #Set fit range to user if specified
        if x_range is not None:
            self.x_range = x_range
        if y_range is not None:
            self.y_range = y_range
           
        off_min = pars[0].mean-error_range*pars[0].std
        off_max = pars[0].mean+error_range*pars[0].std
        off_step = (off_max-off_min)/50.
       
        slope_min = pars[1].mean-error_range*pars[1].std
        slope_max = pars[1].mean+error_range*pars[1].std
        slope_step = (slope_max-slope_min)/50.
        
        o = pars[0].mean if not randomize else np.random.uniform(off_min, off_max, 1)
        oe = pars[0].std if not randomize else np.random.uniform(0, error_range*pars[0].std , 1)
        s = pars[1].mean if not randomize else np.random.uniform(slope_min, slope_max, 1)
        se = pars[1].std if not randomize else np.random.uniform(0, error_range*pars[1].std , 1)
        c = dataset.fit_pcorr[-1][0][1] if not randomize else np.random.uniform(-1, 1, 1)   
        
        if show_errors:
            @interact(offset=(off_min, off_max, off_step),
                      offset_err = (0, error_range*pars[0].std, error_range*pars[0].std/50.),
                      slope=(slope_min, slope_max, slope_step),
                      slope_err = (0, error_range*pars[1].std, error_range*pars[1].std/50.),
                      correlation = (-1,1,0.05)                 
                     )
       
            def update(offset=o, offset_err=oe, slope=s,
                       slope_err=se, correlation=c):  
            
                fig = plt.figure(figsize=(self.dimensions_px[0]/self.screen_dpi,
                                         self.dimensions_px[1]/self.screen_dpi))
            
                ax = fig.add_axes([0,0,1.,1.])
            
                xvals = np.linspace(self.x_range[0], self.x_range[1], 20)
            
                omes = qe.Measurement(offset,offset_err, name="offset")
                smes = qe.Measurement(slope,slope_err, name="slope")
                omes.set_correlation(smes,correlation)
            
                recall = qe.Measurement.minmax_n
                qe.Measurement.minmax_n=1
                fmes = omes + smes*xvals
                qe.Measurement.minmax_n=recall
                fvals = fmes.means
                ferr = fmes.stds
            
                fmax = fvals + ferr
                fmin = fvals - ferr
            
                ax.errorbar(dataset.xdata, dataset.ydata,
                    xerr=dataset.xerr,yerr=dataset.yerr,
                    fmt='o',color=color,markeredgecolor = 'none',
                    label=dataset.name)
            
                ax.plot(xvals,fvals, color=color, label ="linear fit")
                ax.fill_between(xvals, fmin, fmax, facecolor=color,
                                                alpha=0.3, edgecolor = 'none',
                                                interpolate=True)
            
            
                start_x = 0.99 + self.fit_results_x_offset
                start_y = 0.99 + self.fit_results_y_offset
           
                #calculate chi2
                xdata = dataset.xdata
                ydata = dataset.ydata
                yerr = dataset.yerr
                ymodel = offset+slope*xdata
                yres = ydata-ymodel
                chi2 = 0
                ndof = 0
                for i in range(xdata.size):
                    if yerr[i] != 0:
                        chi2 += (yres[i]/yerr[i])**2
                        ndof += 1
                ndof -= 3 #2 parameters, -1
            
                textfit=str(omes)+"\n"+str(smes)+\
                        ("\n chi2/ndof: {:.3f}/{}".format(chi2, ndof) if show_chi2 else "")
               
                ax.annotate(textfit,
                        xy=(start_x, start_y), xycoords = 'axes fraction',
                        fontsize=q.settings["plot_fig_fitres_ftsize"],
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
  
                ax.axis([self.x_range[0], self.x_range[1], 
                                         self.y_range[0], self.y_range[1]])
                ax.set_xlabel(self.labels['xtitle'],
                          fontsize=q.settings["plot_fig_xtitle_ftsize"])
                ax.set_ylabel(self.labels['ytitle'],
                          fontsize=q.settings["plot_fig_ytitle_ftsize"])
                ax.set_title(self.labels['title'],
                          fontsize=q.settings["plot_fig_title_ftsize"])
                ax.legend(loc=self.mpl_legend_location,
                      fontsize = q.settings["plot_fig_leg_ftsize"])
                ax.grid()
                plt.show()
        else: #no errors
            @interact(offset=(off_min, off_max, off_step),
                      slope=(slope_min, slope_max, slope_step)       
                     )
       
            def update(offset=o, slope=s):  
            
                fig = plt.figure(figsize=(self.dimensions_px[0]/self.screen_dpi,
                                         self.dimensions_px[1]/self.screen_dpi))
            
                ax = fig.add_axes([0,0,1.,1.])
            
                xvals = np.linspace(self.x_range[0], self.x_range[1], 20)
                fvals = offset+slope*xvals
   
                ax.errorbar(dataset.xdata, dataset.ydata,
                    xerr=dataset.xerr,yerr=dataset.yerr,
                    fmt='o',color=color,markeredgecolor = 'none',
                    label=dataset.name)
            
                ax.plot(xvals,fvals, color=color, label ="linear fit")

                start_x = 0.99 + self.fit_results_x_offset
                start_y = 0.99 + self.fit_results_y_offset
                
                #calculate chi2
                xdata = dataset.xdata
                ydata = dataset.ydata
                yerr = dataset.yerr
                ymodel = offset+slope*xdata
                yres = ydata-ymodel
                chi2 = 0
                ndof = 0
                for i in range(xdata.size):
                    if yerr[i] != 0:
                        chi2 += (yres[i]/yerr[i])**2
                        ndof += 1
                ndof -= 3 #2 parameters, -1
           
                textfit="offset = {:.2f}\nslope = {:.2f}".format(offset, slope)+\
                        ("\n chi2/ndof: {:.3f}/{}".format(chi2, ndof) if show_chi2 else "")
               
                ax.annotate(textfit,
                        xy=(start_x, start_y), xycoords = 'axes fraction',
                        fontsize=q.settings["plot_fig_fitres_ftsize"],
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
  
                ax.axis([self.x_range[0], self.x_range[1], 
                                         self.y_range[0], self.y_range[1]])
                ax.set_xlabel(self.labels['xtitle'],
                          fontsize=q.settings["plot_fig_xtitle_ftsize"])
                ax.set_ylabel(self.labels['ytitle'],
                          fontsize=q.settings["plot_fig_ytitle_ftsize"])
                ax.set_title(self.labels['title'],
                          fontsize=q.settings["plot_fig_title_ftsize"])
                ax.legend(loc=self.mpl_legend_location,
                      fontsize = q.settings["plot_fig_leg_ftsize"])
                ax.grid()
                plt.show()   

###Some wrapped matplotlib functions
    def mpl_plot(self, *args, **kwargs):
        '''Wrapper for matplotlib plot(), typically to plot a line'''
        if not hasattr(self, 'mplfigure'):
            self.initialize_mpl_figure()
            
        plt.plot(*args, **kwargs)
        
    def mpl_error_bar(self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, 
                      elinewidth=None, capsize=None, barsabove=False, lolims=False,
                      uplims=False, xlolims=False, xuplims=False, errorevery=1,
                      capthick=None, hold=None, data=None, **kwargs):
        '''Wrapper for matplotlib error_bar(), adds points with error bars '''
        if not hasattr(self, 'mplfigure'):
            self.initialize_mpl_figure()
            
        plt.error_bar(self, x, y, yerr, xerr, fmt, ecolor, 
                      elinewidth, capsize, barsabove, lolims,
                      uplims, xlolims, xuplims, errorevery,
                      capthick, hold, data, **kwargs)
        
    def mpl_hist(self,x, bins=10, range=None, normed=False, weights=None,
                 cumulative=False, bottom=None, histtype='bar', align='mid',
                 orientation='vertical', rwidth=None, log=False, color=None,
                 label=None, stacked=False, hold=None, data=None, **kwargs):
        '''Wrapper for matplotlib hist(), creates a histogram'''
        if not hasattr(self, 'mplfigure'):
            self.initialize_mpl_figure()
            
        plt.hist(x, bins, range, normed, weights, cumulative, bottom,
                histtype, align,   orientation, rwidth, log, color,
                label, stacked, hold, data, **kwargs)
            
            
###############################################################################
# Methods for Returning or Rendering Bokeh
###############################################################################    

    def set_bokeh_output(self, output='inline'):
        '''Choose where to output (in a notebook or to a file)'''
        
        if output == 'file' or not qu.in_notebook():
            bi.output_file(self.save_filename,
                           title=self.labels['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            # This must be the first time calling output_notebook,
            # keep track that it's been called:
            qu.bokeh_ouput_notebook_called = True
        else:
            pass        
        
    def populate_bokeh_figure(self):  
        '''Main method for building the plot - this creates the Bokeh figure,
        and then loops through all datasets (and their fit functions), as
        well as user-specified functions, and adds them to the bokeh figure'''
        
        #create a new bokeh figure
        
        #expand the y-range to accomodate the fit results text
        yrange_recall = self.y_range[1]
     
        if self.show_fit_results:
            pixelcount = 0
            for dataset in self.datasets:
                if dataset.nfits > 0:
                    pixelcount += dataset.fit_npars[-1] * 25
            self.y_range[1] += pixelcount * self.y_range[1]/self.dimensions_px[1]
            
            
        self.initialize_bokeh_figure(residuals=False)
        self.y_range[1] = yrange_recall
        
        # create the one for residuals if needed
        if self.show_residuals:
            self.initialize_bokeh_figure(residuals=True)
                              
        #plot the datasets and their latest fit
        legend_offset=0
        self.check_datasets_color_array()
        for dataset, color in zip(self.datasets, self.datasets_colors):
            self.bk_plot_dataset(dataset, residual=False, color=color, show_fit_function=True)
            if dataset.nfits>0:      
                if self.show_fit_results:
                    legend_offset = self.bk_plot_fit_results_text_box(dataset, legend_offset)
                    legend_offset += 3
                if self.show_residuals:
                    self.bk_plot_dataset(dataset, residual=True, color=color)

        #Now add any user defined functions:
        #The range over which to plot the functions:
        if type(self.x_range[0]) in CONSTANT:
            xvals = [self.x_range[0]+self.x_range_margin, 
                     self.x_range[1]-self.x_range_margin]
        else:
            xvals = [0, len(self.x_range)]

        self.check_user_functions_color_array()
        for func, pars, fname, color, sigmas in zip(self.user_functions,
                                                    self.user_functions_pars, 
                                                    self.user_functions_names,
                                                    self.user_functions_colors,
                                                    self.user_functions_sigmas):
        
            self.bk_plot_function(function=func, xdata=xvals,pars=pars, n=q.settings["plot_fcn_npoints"],
                               legend_name= fname, color=color, sigmas=sigmas)

        # Adds lines to the plot
        if self.lines['x'] or self.lines['y']:
            self.bk_add_lines(self.lines)

        #Specify the location of the legend (must be done after stuff has been added)
        self.bkfigure.legend.location = self.bk_legend_location
        self.bkfigure.legend.orientation = self.bk_legend_orientation
        
        if self.show_residuals:
            # Gridplot moved modules, as far as we can tell
            if (hasattr(bi, 'gridplot')):
              self.bkfigure = bi.gridplot([[self.bkfigure], [self.bkres]])
            else:
              self.bkfigure = bl.gridplot([[self.bkfigure], [self.bkres]])
          
        return self.bkfigure
    
    def initialize_bokeh_figure(self, residuals=False):  
        '''Create the bokeh figure with desired labeling and axes'''
        if residuals==False:
            self.bkfigure = bp.figure(
                tools='save, pan, box_zoom, wheel_zoom, reset',
                toolbar_location="above",
                width=self.dimensions_px[0], height=self.dimensions_px[1],
                y_axis_type=self.axes['yscale'],
                y_range=self.y_range,
                x_axis_type=self.axes['xscale'],
                x_range=self.x_range,
                title=self.labels['title'],
                x_axis_label=self.labels['xtitle'],
                y_axis_label=self.labels['ytitle'],
            )
            return self.bkfigure
        else:
            self.set_yres_range_from_fits()
            self.bkres =  bp.figure(
                width=self.dimensions_px[0], height=self.dimensions_px[1]//3,
                tools='save, pan, box_zoom, wheel_zoom, reset',
                toolbar_location="above",
                y_axis_type='linear',
                y_range=self.yres_range,
                x_range=self.bkfigure.x_range,
                x_axis_label=self.labels['xtitle'],
                y_axis_label='Residuals'
            )
            return self.bkres
        
    def bk_add_lines(self, lines):
        '''Adds vertical and horizontal lines to a Bokeh plot.'''
        if not hasattr(self, 'bkfigure'):
            self.bkfigure = self.initialize_bokeh_figure(residuals=False)

        for x_data in lines['x']:
            dashed = 'dashed' if x_data['dashed'] else 'solid'
            self.bkfigure.line([x_data['pos']]*2, self.y_range, line_color=x_data['color'], line_dash=dashed)

        for y_data in lines['y']:
            dashed = 'dashed' if y_data['dashed'] else 'solid'
            self.bkfigure.line(self.x_range, [y_data['pos']]*2, line_color=y_data[color], line_dash=dashed)

    def bk_plot_fit_results_text_box(self, dataset, yoffset=0):
        '''Add a text box with the fit parameters from the last fit to
        the data set'''
        if not hasattr(self, 'bkfigure'):
            self.bkfigure = self.initialize_bokeh_figure(residuals=False)
            
        offset = yoffset    
        start_x = self.dimensions_px[0]-5 + self.fit_results_x_offset   
        start_y = self.dimensions_px[1]-30-offset + self.fit_results_y_offset 
        
        for i in range(dataset.fit_npars[-1]):
            #shorten the name of the fit parameters
            short_name =  dataset.fit_pars[-1][i].__str__().split('_')
            short_name = short_name[0]+"_"+short_name[-1]
            if i > 0:
                offset += 18
            tbox = mo.Label(x=start_x, y=start_y-offset,
                                text_align='right',
                                text_baseline='top',
                                text_font_size='11pt',
                                x_units='screen',
                                y_units='screen',
                                text=short_name,
                                render_mode='css',
                                background_fill_color='white',
                                background_fill_alpha=0.7)
            self.bkfigure.add_layout(tbox)
        return offset
        
    def bk_plot_dataset(self, dataset, residual=False, color='black', show_fit_function=True, fit_index = -1):
        '''Add a dataset to the bokeh figure for the plot - it is better to 
        use add_dataset() to add a dataset to the Plot object and let
        populate_bokeh_figure take care of calling this function'''
        
        index = fit_index if fit_index < dataset.nfits else -1

        if residual == True:
            if not hasattr(self, 'bkres'):
                self.bkres = self.initialize_bokeh_figure(residuals=True)
            return qpu.bk_plot_dataset(self.bkres, dataset, residual=True, color=color, fit_index = index)
            
        if not hasattr(self, 'bkfigure'):
            self.bkfigure = self.initialize_bokeh_figure(residuals=False)

        if dataset.is_histogram:
            if hasattr(dataset, 'hist_data'):
                hist, bins = np.histogram(dataset.hist_data, dataset.bins)
                self.bkfigure.quad(top=hist, bottom=0, left=bins[:-1], right=bins[1:],
                        color=color, alpha=0.7, legend=dataset.name)
            else:
                width = dataset.xdata[-1]-dataset.xdata[-2]
                self.bkfigure.quad(top=dataset.ydata, bottom=0, left=dataset.xdata-width/2,
                        right = dataset.xdata+width/2, color=color, alpha=0.7, legend=dataset.name)
        else:  
            qpu.bk_plot_dataset(self.bkfigure, dataset, residual=False, color=color)
        if dataset.nfits > 0 and show_fit_function:
            self.bk_plot_function(function=dataset.fit_function[index], xdata=dataset.xdata,
                                pars=dataset.fit_pars[index], n=q.settings["plot_fcn_npoints"],
                                legend_name=dataset.name+"_"+dataset.fit_function_name[index],
                                color=color if dataset.fit_color[index] is None else dataset.fit_color[index],
                                sigmas=dataset.xyfitter[index].sigmas)
    
    def bk_add_points_with_error_bars(self, xdata, ydata, xerr=None, yerr=None,
                                   color='black', data_name='dataset'):
        '''Add a set of data points with error bars to the main figure -it is better 
        to use add_dataset if the data should be treated as a dataset that can be fit'''
        if not hasattr(self, 'bkfigure'):
            self.bkfigure = self.initialize_bokeh_figure(residuals=False)
        return qpu.bk_add_points_with_error_bars(self.bkfigure, xdata, ydata, xerr=xerr,
                                              yerr=yerr, color=color,
                                              data_name=data_name)
    
    def bk_plot_function(self, function, xdata, pars=None, n=100,
                      legend_name=None, color='black', sigmas=None):
        '''Add a function to the main figure. It is better to use add_function() and to
        let populate_bokeh_plot() actually add the function.
        
        The function can be either f(x) or f(x, *pars), in which case, if *pars is
        a Measurement_Array, then error bands will be drawn
        '''
        if sigmas == None:
            sigmas = self.errorband_sigma

        if not hasattr(self, 'bkfigure'):
            self.bkfigure = self.initialize_bokeh_figure(residuals=False)
        return qpu.bk_plot_function(self.bkfigure, function, xdata, pars=pars, n=n,
                      legend_name=legend_name, color=color, sigmas=sigmas)       
        
    def bk_show_linear_fit(self, output='inline'):
        '''Fits the last dataset to a linear function and displays the
        result. The fit parameters are not displayed as this function is 
        designed to be used in conjunction with bk_interarct_linear_fit()'''
        
        
        if len(self.datasets) >1:
            print("Warning: only using the last added dataset, and clearing previous fits")
                     
        dataset = self.datasets[-1]
        color = self.datasets_colors[-1]
        
        dataset.clear_fits()
        dataset.fit("linear")
        
        func = dataset.fit_function[-1]
        pars = dataset.fit_pars[-1]
        fname = "linear"       
        
        #Extend the x range to 0
        if self.x_range[0] > -0.5:
            self.x_range[0] = -0.5
            self.y_range[0] = dataset.fit_function[-1](self.x_range[0], *pars.means)
        
        self.bkfigure = self.initialize_bokeh_figure(residuals=False)
        
        self.bk_plot_dataset(dataset, residual=False,color=color, show_fit_function=False)
        
        xvals = [self.x_range[0]+self.x_range_margin, 
                 self.x_range[1]-self.x_range_margin]
        
        line, patches = self.bk_plot_function( function=func, xdata=xvals,
                              pars=pars, n=100, legend_name= fname,
                              color=color)
        
        #stuff that is only needed by interact_linear_fit
        self.linear_fit_line = line
        self.linear_fit_patches = patches
        self.linear_fit_pars = pars
        self.linear_fit_corr = dataset.fit_pcorr[-1][0][1]
               
        #Specify the location of the legend
        self.bkfigure.legend.location = self.bk_legend_location      
        self.show(output=output,populate_figure=False)

    def bk_interact_linear_fit(self, error_range = 2):  
        '''After show_linear_fit() has been called, this will display
        sliders allowing the user to adjust the parameters of the linear
        fit - only works in a notebook, require ipywigets'''
        
        if not qu.in_notebook():
            print("Can only use this feature in a notebook, sorry")
            return
        
        
        off_mean = self.linear_fit_pars[0].mean
        off_std = self.linear_fit_pars[0].std
        off_min = off_mean-error_range*off_std
        off_max = off_mean+error_range*off_std
        off_step = (off_max-off_min)/50.
       
        slope_mean = self.linear_fit_pars[1].mean
        slope_std = self.linear_fit_pars[1].std
        slope_min = slope_mean-error_range*slope_std
        slope_max = slope_mean+error_range*slope_std
        slope_step = (slope_max-slope_min)/50.
        
            
        @interact(offset=(off_min, off_max, off_step),
                  offset_err = (0, error_range*off_std, error_range*off_std/50.),
                  slope=(slope_min, slope_max, slope_step),
                  slope_err = (0, error_range*slope_std, error_range*slope_std/50.),
                  correlation = (-1,1,0.05)                 
                 )
        def update(offset=off_mean, offset_err=off_std, slope=slope_mean, slope_err=slope_std, correlation=self.linear_fit_corr):

            recall = qe.Measurement.minmax_n
            qe.Measurement.minmax_n=1
            omes = qe.Measurement(offset,offset_err)
            smes = qe.Measurement(slope,slope_err)
            omes.set_correlation(smes,correlation)
            xdata = np.array(self.linear_fit_line.data_source.data['x'])
            fmes = omes+ smes*xdata
            qe.Measurement.minmax_n=recall
            
            ymax = fmes.means+fmes.stds
            ymin = fmes.means-fmes.stds      
            
            self.linear_fit_line.data_source.data['y'] = fmes.means
            self.linear_fit_patches.data_source.data['y'] = np.append(ymax,ymin[::-1])

            bi.push_notebook()
