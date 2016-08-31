import scipy.optimize as sp
import numpy as np
import qexpy.error as qe
import qexpy.utils as qu
import qexpy.fitting as qf

from math import pi
import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.models as mo

CONSTANT = qu.number_types
ARRAY = qu.array_types


class Plot:
    '''Objects which contain a dataset and any number of fuctions which can
    be shown on a Bokeh plot

    Plot objects are indirectly dependent on Bokeh plots. Plot objects
    contain data and associated error lists, as well as functions or fits
    of the data. These points are used to contruct Bokeh plots by using
    custom functions to draw a series of circles, lines, and rectangles
    which mimic errorbars and data points.
    '''

    def __init__(self, x, y, xerr=None, yerr=None, data_name=None):
        '''
        Constructor requiring two measeurement objects, or lists of data and
        error values.

        Plotting objects do not create Bokeh objects until shown or if the
        Bokeh object is otherwise requested. Class methods built here are
        used to record the data to be plotted and track what the user has
        requested to be plotted.
        '''
        
        self.datasets=[]
        self.datasets.append(qf.XYDataSet(x, y, xerr=xerr, yerr=yerr, data_name=data_name))
        
        self.xunits = self.datasets[-1].xunits
        self.xname = self.datasets[-1].xname
        
        self.yunits = self.datasets[-1].yunits
        self.yname = self.datasets[-1].yname    
        
        self.colors = {
            'Data Points': ['red', 'black'],
            'Fit': ['blue', 'green'],
            'Function': ['orange', 'navy'],
            'Error': 'red'}
        
        self.plot_para = {
            'xscale': 'linear', 'yscale': 'linear', 'filename': 'Plot'}
        
        self.flag = {'fitted': False, 'residuals': False,
                     'Manual': False} # analysis:ignore
        self.attributes = {
            'title': self.yname+' versus '+self.xname,
            'xaxis': self.xname+'/'+self.xunits,
            'yaxis': self.yname+'/'+self.yunits,
            'data': data_name, 'function': (), }
                
        self.x_range = self.datasets[-1].get_x_range(2)
        self.y_range = self.datasets[-1].get_y_range(2)
        
        self.dimensions = [600, 400]
        self.sigma = 1
        
    def fit(self, model=None, parguess=None, fit_range=None, datasetindex=-1):
        return self.datasets[datasetindex].fit()
        
###############################################################################
# User Methods for adding to Plot Objects
###############################################################################

    def residuals(self):
        '''Request residual output for plot.'''
        if self.datasets[-1].nfits>0:
            self.flag['residuals'] = True

    
    def add_function(self, function):
        '''Adds a specified function to the list of functions to be plotted.

        Functions are only plotted when a Bokeh object is created, thus user
        specified functions are stored to be plotted later.
        '''
        
        #check if we should change the y-axis range to accomodate the function
        f = function(datasets[-1].xdata)
        fmax = f.max()
        fmin = f.min()
        if fmax > self.yrange[1]:
            self.yrange[1]=fmax
        if fmin < self.yrange[0]:
            self.yrange[0]=fmin
            
        self.attributes['function'] += (function,)

# This should be handled by changing the dataset
#    def manual_errorbar(self, data, function):
#        '''Manually specify the location of a datapoint with errorbars.'''
#        import qexpy.error_operations as op
#        data, function = op.check_values(data, function)
#        self.manual_data = (data, function(data))
#        self.flag['Manual'] = True

###############################################################################
# Methods for changing parameters of Plot Object
###############################################################################
    def add_dataset(self, dataset):
        self.datasets.append(dataset)
    
    def set_plot_range(self, x_range=None, y_range=None):
        if type(x_range) in ARRAY and len(x_range) is 2:
            self.x_range = x_range
        elif x_range is not None:
            print('''X range must be a list containing a minimun and maximum
            value for the range of the plot.''')

        if type(y_range) in ARRAY and len(y_range) is 2:
            self.y_range = y_range
        elif y_range is not None:
            print('''Y range must be a list containing a minimun and maximum
            value for the range of the plot.''')

    def set_colors(self, data=None, error=None, fit=None):
        '''Method to changes colors of data or function lines.

        User can specify a list or tuple of color strings for the data points
        or functions, as multiple datasets and functions will be plotted with
        different functions.
        '''
        if data is not None:
            try:
                len(data)
            except TypeError:
                self.colors['Data Points'][0] = data
            else:
                self.colors['Data Points'] = data

        if error is not None:
            self.colors['Error'] = error

        if type(fit) is str:
            self.colors['Fit'] = fit

    def set_name(self, title=None, xlabel=None, ylabel=None, data_name=None, ):
        '''Change the labels for plot axis, datasets, or the plot itself.

        Method simply overwrites the automatically generated names used in
        the Bokeh plot.'''
        if title is not None:
            self.attributes['title'] = title

        if xlabel is not None:
            self.attributes['xname'] = xlabel

        if ylabel is not None:
            self.attributes['yname'] = ylabel

        if data_name is not None:
            self.attributes['Data'] = data_name

    def resize_plot(self, width=None, height=None):
        if width is None:
            width = 600
        if height is None:
            height = 400
        self.dimensions = [width, height]

    def set_errorband_sigma(self, sigma=1):
        '''Change the confidence bounds of the error range on a fit.
        '''
        self.sigma = sigma

###############################################################################
# Methods for Returning or Rendering Bokeh
###############################################################################

    def print_fit_parameters(self):
        if self.datasets[-1].nfits>0:
            print("Fit parameters:\n"+self.datasets[-1].fit_pars[-1])
            
    def get_bokeh_figure(self):
        #disable MinMax to speed things up
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1
        
        # create a new plot
        self.figure = bp.figure(
            tools='save, pan, box_zoom, wheel_zoom, reset',
            width=self.dimensions[0], height=self.dimensions[1],
            y_axis_type=self.plot_para['yscale'],
            y_range=self.y_range,
            x_axis_type=self.plot_para['xscale'],
            x_range=self.x_range,
            title=self.attributes['title'],
            x_axis_label=self.attributes['xaxis'],
            y_axis_label=self.attributes['yaxis'],
        )
        # creat the one for residuals if needed
        if self.flag['residuals']:
            self.res = bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                tools='save, pan, box_zoom, wheel_zoom, reset',
                y_axis_type='linear',
                y_range=self.datasets[-1].get_yres_range(),
                x_range=self.figure.x_range,
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )
                    
            
        #plot the data sets and their latest fit
        count = 0
        for dataset in self.datasets:
            plot_dataset(self.figure, dataset, residual=False,
                         data_color=self.colors['Data Points'][count])
                   
            if dataset.nfits>0:
                plot_function(self.figure, function=dataset.fit_function[-1], xdata=dataset.xdata,
                              fpars=dataset.fit_pars[-1], n=1000, legend_name=dataset.fit_function_name[-1],
                              color=self.colors['Data Points'][count])
                #Draw fit parameters only for the first dataset
                if count<1:
                     for i in range(dataset.fit_npars[-1]):
                        short_name =  dataset.fit_pars[-1][i].__str__().split('_')
                        short_name = short_name[0]+"_"+short_name[-1]
                        citation = mo.Label(x=590, y=320+20*i,
                                    text_align='right',
                                    text_baseline='top',
                                    text_font_size='11pt',
                                    x_units='screen',
                                    y_units='screen',
                                    text=short_name,
                                    render_mode='css',
                                    background_fill_color='white',
                                    background_fill_alpha=1.0)
                        self.figure.add_layout(citation)
                        
                if self.flag['residuals']:
                    plot_dataset(self.res, dataset, residual=True,
                                 data_color=self.colors['Data Points'][count])
            count += 1
        
        self.figure.legend.location = "top_left"
        
        if self.flag['residuals']:
            self.figure = bi.gridplot([[self.figure], [self.res]])
            
        #TODO: Draw the rest (e.g. functions and error bands)    
    
            
            
        return self.figure
        
    def get_bokeh(self):
        '''Return Bokeh plot object for the plot acted upon.

        If no residual plot exists, a single Bokeh object, containing the
        main plot is returned. Else, a tuple with the main and residual plot
        in that order is returned.'''
        
        #disable MinMax to speed things up
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1
        
        # create a new plot
        self.p = bp.figure(
            tools='save, pan, box_zoom, wheel_zoom, reset',
            width=self.dimensions[0], height=self.dimensions[1],
            y_axis_type=self.plot_para['yscale'],
            y_range=self.y_range,
            x_axis_type=self.plot_para['xscale'],
            x_range=self.y_range,
            title=self.attributes['title'],
            x_axis_label=self.attributes['xaxis'],
            y_axis_label=self.attributes['yaxis'],
        )

        # add datapoints with errorbars
        _error_bar(self)

        if self.flag['Manual'] is True:
            _error_bar(self,
                       xdata=self.manual_data[0], ydata=self.manual_data[1])

        if self.flag['fitted'] is True:
            _plot_function(
                self, self.xdata,
                lambda x: self.fit_function(x, *self.fit_parameters),
                legend_name='Fit')

            self.function_counter += 1

            if self.fit_parameters[1].mean > 0:
                self.p.legend.location = "top_left"
            else:
                self.p.legend.location = "top_right"
        else:
            self.p.legend.location = 'top_right'

        for func in self.attributes['function']:
            _plot_function(
                self, self.xdata, func)
            self.function_counter += 1

        if self.flag['residuals'] is False:
            qe.Measurement.minmax_n=recall
            return self.p
        else:
            self.res = bp.figure(
                tools='save, pan, box_zoom, wheel_zoom, reset',
                width=self.dimensions[0], height=self.dimensions[1]//3,
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=[min(self.xdata)-2*max(self.xerr),
                         max(self.xdata)+2*max(self.xerr)],
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(self, residual=True)
            qe.Measurement.minmax_n=recall
            return (self.p, self.res)
       

    def show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods simply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        #disable MinMax to speed things up
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1
        
        if output == 'file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            # This must be the first time calling output_notebook,
            # keep track that it's been called:
            qu.bokeh_ouput_notebook_called = True

        # create a new plot
        self.p = bp.figure(
            width=self.dimensions[0], height=self.dimensions[1],
            toolbar_location='above',
            tools='save, pan, box_zoom, wheel_zoom, reset',
            y_axis_type=self.plot_para['yscale'],
            y_range=self.y_range,
            x_axis_type=self.plot_para['xscale'],
            x_range=self.x_range,
            title=self.attributes['title'],
            x_axis_label=self.attributes['xaxis'],
            y_axis_label=self.attributes['yaxis'],
        )

        # add datapoints with errorbars
        _error_bar(self)

        if self.flag['Manual'] is True:
            _error_bar(self,
                       xdata=self.manual_data[0], ydata=self.manual_data[1])

        if self.flag['fitted'] is True:
            data = [min(self.xdata)-max(self.xerr),
                    max(self.xdata)+max(self.xerr)]
            _plot_function(
                self, data,
                lambda x: self.fit_function(x, *self.fit_parameters),
                color=self.colors['Fit'][0])

            for i in range(len(self.fit_parameters)):
                citation = mo.Label(x=590, y=320+20*i,
                                    text_align='right',
                                    text_baseline='top',
                                    text_font_size='11pt',
                                    x_units='screen',
                                    y_units='screen',
                                    text=self.fit_parameters[i].__str__(),
                                    render_mode='css',
                                    background_fill_color='white',
                                    background_fill_alpha=1.0)
                self.p.add_layout(citation)

            self.function_counter += 1

            if self.fit_parameters[1].mean > 0:
                self.p.legend.location = "top_left"
            else:
                self.p.legend.location = "top_right"
        else:
            self.p.legend.location = 'top_right'

        for func in self.attributes['function']:
            _plot_function(
                self, self.xdata, func)
            self.function_counter += 1

        if self.flag['residuals'] is False:
            bp.show(self.p)
            qe.Measurement.minmax_n=recall
            return self.p
        else:

            self.res = bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                tools='save, pan, box_zoom, wheel_zoom, reset',
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=self.p.x_range,
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(self, residual=True)

            gp_alt = bi.gridplot([[self.p], [self.res]])
            bp.show(gp_alt)
            qe.Measurement.minmax_n=recall
            return gp_alt

    def show_on(self, plot2, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        #disable MinMax to speed things up
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1

        if output == 'file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            # This must be the first time calling output_notebook,
            # keep track that it's been called:
            qu.bokeh_ouput_notebook_called = True

        if min(plot2.xdata) < self.y_range[0]:
            self.y_range[0] = min(plot2.xdata)

        if max(plot2.xdata) > self.y_range[1]:
            self.y_range[1] = max(plot2.xdata)

        # create a new plot
        self.p = bp.figure(
            width=self.dimensions[0], height=self.dimensions[1],
            toolbar_location='above',
            tools='save, pan, box_zoom, wheel_zoom, reset',
            y_axis_type=self.plot_para['yscale'],
            #  y_range=self.y_range,
            x_axis_type=self.plot_para['xscale'],
            #  x_range=self.x_range,
            title=self.attributes['title'],
            x_axis_label=self.attributes['xaxis'],
            y_axis_label=self.attributes['yaxis'],
        )

        # add datapoints with errorbars
        _error_bar(self)
        _error_bar(plot2, plot_object=self, color=1)

        if self.flag['Manual'] is True:
            _error_bar(self,
                       xdata=self.manual_data[0], ydata=self.manual_data[1])

        if plot2.flag['Manual'] is True:
            _error_bar(plot2,
                       xdata=plot2.manual_data[0], ydata=plot2.manual_data[1],
                       plot_object=self)

        if self.flag['fitted'] is True:
            
            data = [min(self.xdata)-max(self.xerr),
                    max(self.xdata)+max(self.xerr)]
            _plot_function(
                self, data,
                lambda x: self.fit_function(x, *self.fit_parameters),
                legend_name='Fit', color=plot2.colors['Fit'][0])
            
            self.function_counter += 1

            if self.fit_parameters[1].mean > 0:
                self.p.legend.location = "top_left"
            else:
                self.p.legend.location = "top_right"
        else:
            self.p.legend.location = 'top_right'

        if plot2.flag['fitted'] is True:
            data = [min(plot2.xdata)-max(plot2.xerr),
                    max(plot2.xdata)+max(plot2.xerr)]
            _plot_function(
                self, data,
                lambda x: plot2.fit_function(x, *plot2.fit_parameters),
                legend_name='Second Fit', color=plot2.colors['Fit'][1])
            
            self.function_counter += 1

        for func in self.attributes['function']:
            _plot_function(
                self, self.xdata, func)
            self.function_counter += 1

        if self.flag['residuals'] is False and\
                plot2.flag['residuals'] is False:
            bp.show(self.p)
            qe.Measurement.minmax_n = recall
            return self.p

        elif self.flag['residuals'] is True and\
                plot2.flag['residuals'] is True:

            self.res = bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=self.p.x_range,
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals',
            )

            # plot y errorbars
            _error_bar(self, residual=True)

            plot2.res = bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                y_axis_type='linear',
                y_range=[min(plot2.yres)-2*max(plot2.yerr),
                         max(plot2.yres)+2*max(plot2.yerr)],
                x_range=self.p.x_range,
                x_axis_label=plot2.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(plot2, residual=True, color=1)

            gp_alt = bi.gridplot([[self.p], [self.res], [plot2.res]])
            bp.show(gp_alt)
            qe.Measurement.minmax_n = recall
            return gp_alt

###############################################################################
# First Year Methods
###############################################################################

    def first_year_fit(self, model=None, guess=None, fit_range=None):
        '''Fit data, by least squares method, to a model. Model must be
        provided or specified from built in models. User specified models
        require and inital guess for the fit parameters.

        By default a linear fit is used, if the user enters a string
        for another fit model which is built-in, that model is used.
        If the user provides a fit function, with two arguments, the first
        for the independent variable, and the second for the list of
        parameters, an inital guess of the parameters in the form of a
        list of values must be provided.
        '''
        import numpy as np

        linear = ('linear', 'Linear', 'line', 'Line',)
        gaussian = ('gaussian', 'Gaussian', 'Gauss', 'gauss', 'normal',)
        exponential = ('exponential', 'Exponential', 'exp', 'Exp',)

        if guess is None:
            if model in linear:
                guess = [1, 1]
            elif model[0] is 'p':
                degree = int(model[len(model)-1]) + 1
                guess = [1]*degree
            elif model in gaussian:
                guess = [1]*2
            elif model in exponential:
                guess = [1]*2

        if model is not None:
            if type(model) is not str:
                self.flag['Unknown Function'] = True
                self.fit_method = None
                self.fit_function = model

            elif model in linear:
                self.fit_method = 'linear'
                self.fit_function = Plot.fits['linear']

            else:
                raise TypeError('''Input must be string, either 'linear',
                                'gaussian', 'exponential', 'polyn' for a
                                polynomial of order n, or a custom
                                function.''')

        else:
            print('Using a linear fit by default.')
            self.fit_method = 'linear'
            self.fit_function = Plot.fits[model]

            def model(x, *pars):
                return self.fit_function(x, *pars)

        pars_guess = guess

        if fit_range is None:
            data_range = self.xdata
        elif type(fit_range) in ARRAY and len(fit_range) is 2:
            data_range = []
            for i in self.xdata:
                if i >= min(fit_range) and i <= max(fit_range):
                    data_range.append(i)

        self.pars_fit, self.pcov = sp.curve_fit(
                                    model, data_range, self.ydata,
                                    sigma=self.yerr, p0=pars_guess)
        self.pars_err = np.sqrt(np.diag(self.pcov))

        if self.xerr is not None:
            yerr_eff = np.power(
                (np.power(self.yerr, 2) +
                np.power(np.multiply(self.xerr, #analysis:ignore
                num_der(lambda x: model(x, *self.pars_fit) , #analysis:ignore
                self.xdata)), 2)), 0.5) #analysis:ignore

            self.pars_fit, self.pcov = sp.curve_fit(
                                        model, data_range, self.ydata,
                                        sigma=yerr_eff, p0=pars_guess)
            self.pars_err = np.sqrt(np.diag(self.pcov))

        for i in range(len(self.pars_fit)):
            if self.fit_method is 'gaussian':
                if i is 0:
                    name = 'mean'
                elif i is 1:
                    name = 'standard deviation'
            elif self.fit_method is 'linear':
                if i is 0:
                    name = 'intercept'
                elif i is 1:
                    name = 'slope'
            else:
                name = 'par %d' % (i)

            self.fit_parameters += (
                qe.Measurement(self.pars_fit[i], self.pars_err[i], name=name),)
        self.flag['fitted'] = True

    def first_year_show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''

        if output == 'file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            # This must be the first time calling output_notebook,
            # keep track that it's been called:
            qu.bokeh_ouput_notebook_called = True

        # create a new plot
        self.p = bp.figure(
            width=self.dimensions[0], height=self.dimensions[1],
            toolbar_location='above',
            tools='save, pan, box_zoom, wheel_zoom, reset',
            y_axis_type=self.plot_para['yscale'],
            y_range=self.y_range,
            x_axis_type=self.plot_para['xscale'],
            x_range=self.x_range,
            title=self.attributes['title'],
            x_axis_label=self.attributes['xaxis'],
            y_axis_label=self.attributes['yaxis'],
        )

        # add datapoints with errorbars
        _error_bar(self)

        if self.flag['Manual'] is True:
            _error_bar(self,
                       xdata=self.manual_data[0], ydata=self.manual_data[1])

        if self.flag['fitted'] is True:
            data = [0, max(self.xdata)+max(self.xerr)]
            _plot_function(
                self, data,
                lambda x: self.fit_function(x, *self.pars_fit))

            self.function_counter += 1

            if self.fit_parameters[1].mean > 0:
                self.p.legend.location = "top_left"
            else:
                self.p.legend.location = "top_right"
        else:
            self.p.legend.location = 'top_right'

        for func in self.attributes['function']:
            _plot_function(
                self, self.xdata, func)
            self.function_counter += 1

        if self.flag['residuals'] is False:
            bp.show(self.p)
            return self.p
        else:

            self.res = bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                tools='save, pan, box_zoom, wheel_zoom, reset',
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=self.p.x_range,
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(self, residual=True)

            gp_alt = bi.gridplot([[self.p], [self.res]])
            bp.show(gp_alt)
            return gp_alt


###############################################################################
# Functions for Plotting common objects
###############################################################################

def plot_dataset(figure, dataset, residual=False, data_color='black'):
    '''Given a bokeh figure, this will add data points with errors from the dataset'''
  
    xdata = dataset.xdata
    xerr = dataset.xerr
    
    if residual is True and dataset.nfits>0:
        ydata = dataset.yres[-1].get_means()
        yerr = dataset.yres[-1].get_stds()
    else:
        ydata = dataset.ydata
        yerr = dataset.yerr
     
    add_points_with_error_bars(figure, xdata, ydata, xerr, yerr, data_color, dataset.name)
    
def add_points_with_error_bars(figure, xdata, ydata, xerr=None, yerr=None, data_color='black', data_name='dataset'):
    
    if xdata.size != ydata.size:
        print("Error: x and y data must have the same number of points")
        return None
    
    #Draw points:    
    figure.circle(xdata, ydata, color=data_color, size=2, legend=data_name)

    if isinstance(xerr,np.ndarray) or isinstance(yerr,np.ndarray):
        #Add error bars
        for i in range(xdata.size):
            
            xcentral = [xdata[i], xdata[i]]
            ycentral = [ydata[i], ydata[i]]
            
            #x error bar
            if isinstance(xerr,np.ndarray) and xerr.size == xdata.size and xerr[i]>0:
                xends = [xdata[i]-xerr[i], xdata[i]+xerr[i]]
                figure.line(xends,ycentral, color=data_color)
                #winglets on x error bar:
                figure.rect(x=xends, y=ycentral, height=5, width=0.2,
                    height_units='screen', width_units='screen',
                    color=data_color,legend=data_name)
                
            #y error bar    
            if isinstance(yerr,np.ndarray) and yerr.size == xdata.size and yerr[i]>0:    
                yends = [ydata[i]-yerr[i], ydata[i]+yerr[i]]
                figure.line(xcentral, yends, color=data_color)
                #winglets on y error bar:
                figure.rect(x=xcentral, y=yends, height=0.2, width=5,
                    height_units='screen', width_units='screen',
                    color=data_color,legend=data_name)
    
    
def plot_function(figure, function, xdata, fpars=None, n=1000, legend_name=None, color='black'):
    '''Plot a function evaluated over the range of xdata'''
    xvals = np.linspace(min(xdata), max(xdata), n)
    
    if fpars is None:
        fvals = function(xvals)
    elif isinstance(fpars, qe.Measurement_Array):
        fvals = function(xvals, *(fpars.get_means()))
    elif isinstance(fpars,(list, np.ndarray)):
        fvals = function(xvals, *fpars)
    else:
        pass
    figure.line(xvals, fvals, legend=legend_name, line_color=color)
        

def _plot_function(self, xdata, theory, n=1000, legend_name=None, color=None):
    '''Semi-privite function to plot a function over a given range of values.

    Curves are generated by creating a series of lines between points, the
    parameter n is the number of points. Line color is given by the plot
    attribute containing a list of colors which are assigned uniquly to a
    curve.'''
    xrange = np.linspace(min(xdata), max(xdata), n)
    y_theory = theory(min(xdata))
    y_mid = []

    if type(color) is str:
        func_color = color
    else:
        func_color = self.colors['Function'][self.function_counter]

    try:
        y_theory.type
    except AttributeError:
        for i in range(n):
            y_mid.append(theory(xrange[i]))
        self.fit_line = self.p.line(
            xrange, y_mid, legend=legend_name,
            line_color=func_color)

    else:
        y_max = []
        y_min = []
        for i in range(n):
            y_theory = theory(xrange[i])
            y_mid.append(y_theory.mean)
            y_max.append(y_theory.mean+self.sigma*y_theory.std)
            y_min.append(y_theory.mean-self.sigma*y_theory.std)
        self.fit_line = self.p.line(
            xrange, y_mid, legend=legend_name,
            line_color=func_color)

        xrange_reverse = list(reversed(xrange))
        y_min_reverse = list(reversed(y_min))
        xrange = list(xrange)
        y_max = list(y_max)

        self.fit_range = self.p.patch(
            x=xrange+xrange_reverse, y=y_max+y_min_reverse,
            fill_alpha=0.3,
            fill_color=func_color,
            line_color=func_color,
            line_dash='dashed', line_alpha=0.3,
            legend=legend_name)


###############################################################################
# Miscellaneous Functions
###############################################################################



def update_plot(self):
    ''' Creates interactive sliders in Jupyter Notebook to adjust fit.
    '''
    from ipywidgets import interact

    range_argument = ()
    for par in self.fit_parameters:
        min_val = par.mean - 2*par.std
        increment = (par.mean-min_val)*0.01
        range_argument += (min_val, par.mean, increment)

    for par in self.fit_parameters:
        increment = (par.std)*0.01
        range_argument += (0, par.std, increment)

    @interact(b=self.fit_parameters[0].mean, m=self.fit_parameters[1].mean,
              b_error=self.fit_parameters[0].std,
              m_error=self.fit_parameters[1].std)
    def update(b=self.fit_parameters[0].mean,
               m=self.fit_parameters[1].mean,
               b_error=self.fit_parameters[0].std,
               m_error=self.fit_parameters[1].std):
        import numpy as np

        self.fit_line.data_source.data['y'] = np.multiply(
            m, self.fit_line.data_source.data['x']) + b

        error_top_line = np.multiply(
            m+m_error, self.fit_line.data_source.data['x']) + b+b_error
        error_bottom_line = np.multiply(
            m-m_error, self.fit_line.data_source.data['x']) + b-b_error

        self.fit_range.data_source.data['y'] = list(error_top_line) +\
            list(error_bottom_line)
        bi.push_notebook()


def num_der(function, point, dx=1e-10):
    '''
    Returns the first order derivative of a function.
    Used in combining xerr and yerr.
    '''
    import numpy as np
    point = np.array(point)
    return np.divide(function(point+dx)-function(point), dx)
