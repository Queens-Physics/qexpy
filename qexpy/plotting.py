import scipy.optimize as sp
import numpy as np
import qexpy.error as e
import qexpy.utils as qu
from math import pi
import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.models as mo
from numpy import int64, float64, ndarray, int32, float32

CONSTANT = (int, float, int64, float64, int32, float32)
ARRAY = (list, tuple, ndarray)



class Plot:
    '''Objects which contain a dataset and any number of fuctions which can
    be shown on a Bokeh plot

    Plot objects are indirectly dependent on Bokeh plots. Plot objects
    contain data and associated error lists, as well as functions or fits
    of the data. These points are used to contruct Bokeh plots by using
    custom functions to draw a series of circles, lines, and rectangles
    which mimic errorbars and data points.
    '''

    def polynomial(x, *pars):
        '''Function for a polynomial of nth order, requiring n pars.'''
        poly = 0
        n = 0

        for par in pars:
            poly += np.multiply(par, np.power(x, n))
            n += 1
        return poly

    def gauss(x, mean, std):
        '''Fucntion of gaussian distribution'''
        from error import exp
        return (2*pi*std**2)**(-1/2)*exp(-(x-mean)**2/2/std**2)

    fits = {
            'linear': lambda x, b, m: b+np.multiply(m, x),
            'exponential': lambda x, b, m: np.exp(b+m*x),
            'polynomial': polynomial,
            'gaussian': gauss,
            }

    def __init__(self, x, y, xerr=None, yerr=None, data_name=None):
        '''
        Constructor requiring two measeurement objects, or lists of data and
        error values.

        Plotting objects do not create Bokeh objects until shown or if the
        Bokeh object is otherwise requested. Class methods built here are
        used to record the data to be plotted and track what the user has
        requested to be plotted.
        '''
        self.pars_fit = []
        self.pars_err = []
        self.pcov = []
        data_transform(self, x, y, xerr, yerr)

        self.colors = {
            'Data Points': ['red', 'black'],
            'Fit': ['blue', 'green'],
            'Function': ['orange', 'navy'],
            'Error': 'red'}
        self.fit_method = 'linear'
        self.fit_function = Plot.fits[self.fit_method] # analysis:ignore
        self.plot_para = {
            'xscale': 'linear', 'yscale': 'linear', 'filename': 'Plot'}
        self.flag = {'fitted': False, 'residuals': False,
                     'Manual': False} # analysis:ignore
        self.attributes = {
            'title': self.xname+' versus '+self.yname,
            'xaxis': self.xname+' '+self.xunits,
            'yaxis': self.yname+' '+self.yunits,
            'data': data_name, 'function': (), }
        self.fit_parameters = ()
        self.yres = None
        self.function_counter = 0
        self.manual_data = ()
        self.x_range = [min(self.xdata)-2*max(self.xerr),
                        max(self.xdata)+2*max(self.xerr)]
        self.y_range = [min(self.ydata)-2*max(self.yerr),
                        max(self.ydata)+2*max(self.yerr)]
        self.dimensions = [600, 400]
        self.sigma = 1

###############################################################################
# User Methods for adding to Plot Objects
###############################################################################

    def residuals(self):
        '''Request residual output for plot.'''
        if self.flag['fitted'] is False:
            self.fit(model='linear')
            print('Fit not defined, using linear fit by default.')

        # Calculate residual values
        yfit = self.fit_function(self.xdata, *self.pars_fit)
        # Generate a set of residuals for the fit
        self.yres = self.ydata-yfit

        self.flag['residuals'] = True

    def fit(self, model=None, guess=None, fit_range=None):
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

            elif model[0] is 'p' and model[1] is 'o':
                self.fit_function = Plot.fits['polynomial']

                def model(x, *pars):
                    return self.fit_function(x, *pars)

            elif model in linear:
                self.fit_method = 'linear'
                self.fit_function = Plot.fits['linear']

                def model(x, *pars):
                    return self.fit_function(x, *pars)

            elif model in exponential:
                self.fit_method = 'exponential'
                self.fit_function = Plot.fits['exponential']

                def model(x, *pars):
                    return self.fit_function(x, *pars)

            elif model in gaussian:
                self.fit_method = 'gaussian'
                self.fit_function = Plot.fits['gaussian']

                def model(x, *pars):
                    return self.fit_function(x, *pars)

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

        if self.flag['fitted'] is True:
            print('''A fit of the data already exists, overwriting previous
                  fit.''')
            self.fit_parameters = ()
            self.function_counter -= 1

        pars_guess = guess

        if fit_range is None:
            data_range = self.xdata
        elif type(fit_range) in ARRAY and len(fit_range) is 2:
            data_range = []
            for i in self.xdata:
                if i >= min(fit_range) and i <= max(fit_range):
                    data_range.append(i)

        data_range = np.array(data_range)
        ydata = np.array(self.ydata)
        yerr = np.array(self.yerr)
        self.pars_fit, self.pcov = sp.curve_fit(
                                    model, data_range, ydata,
                                    sigma=yerr, p0=pars_guess)
        self.pars_err = np.sqrt(np.diag(self.pcov))

        # Use derivative method to factor x error into fit
        if self.xerr is not None:
            yerr_eff = np.power(
                (np.power(self.yerr,
                          2) + np.power(np.multiply(self.xerr, num_der
                                                    (lambda x: model
                                                     (x, *self.pars_fit
                                                      ), self.xdata
                                                     )), 2)), 1/2)

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
                e.Measurement(self.pars_fit[i], self.pars_err[i], name=name),)

            for i in range(len(self.fit_parameters)-1):
                self.fit_parameters[0].set_covariance(self.fit_parameters[i+1],
                                                      self.pcov[0][i+1])
        self.flag['fitted'] = True

    def function(self, function):
        '''Adds a specified function to the list of functions to be plotted.

        Functions are only plotted when a Bokeh object is created, thus user
        specified functions are stored to be plotted later.
        '''
        if function(min(self.xdata)) > self.y_range[1] or\
                function(max(self.xdata)) > self.y_range[1]:

            if function(min(self.xdata)) > function(max(self.xdata)):
                self.y_range[1] = function(min(self.xdata))
            elif function(min(self.xdata)) < function(max(self.xdata)):
                self.y_range[1] = function(max(self.xdata))

        if function(min(self.xdata)) < self.y_range[0] or\
                function(max(self.xdata)) < self.y_range[0]:

            if function(min(self.xdata)) < function(max(self.xdata)):
                self.y_range[0] = function(min(self.xdata))
            elif function(min(self.xdata)) > function(max(self.xdata)):
                self.y_range[0] = function(max(self.xdata))

        self.attributes['function'] += (function,)

    def manual_errorbar(self, data, function):
        '''Manually specify the location of a datapoint with errorbars.'''
        import qexpy.error_operations as op
        data, function = op.check_values(data, function)
        self.manual_data = (data, function(data))
        self.flag['Manual'] = True

###############################################################################
# Methods for changing parameters of Plot Object
###############################################################################

    def plot_range(self, x_range=None, y_range=None):
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
        self.dimensions[width, height]

    def set_errorband_sigma(self, sigma=1):
        '''Change the confidance bounds of the error range on a fit.
        '''
        self.sigma = sigma

###############################################################################
# Methods for Returning or Rendering Bokeh
###############################################################################

    def print_fit_parameters(self):
        if self.flag['Fitted'] is False:
            print('''Please create a fit of the data using .fit to find the
            fit parameters.''')
        else:
            for par in self.fit_parameters:
                print(par)

    def get_bokeh(self):
        '''Return Bokeh plot object for the plot acted upon.

        If no residual plot exists, a single Bokeh object, containing the
        main plot is returned. Else, a tuple with the main and residual plot
        in that order is returned.'''
        # create a new plot
        self.p = bp.figure(
            tools='save, pan, box_zoom, wheel_zoom, reset',
            width=self.dimensions[0], height=self.dimensions[1],
            y_axis_type=self.plot_para['yscale'],
            y_range=[min(self.ydata)-2*max(self.yerr),
                     max(self.ydata)+2*max(self.yerr)],
            x_axis_type=self.plot_para['xscale'],
            x_range=[min(self.xdata)-2*max(self.xerr),
                     max(self.xdata)+2*max(self.xerr)],
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
            return (self.p, self.res)

    def show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods simply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        
        if output =='file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            #This must be the first time calling output_notebook,
            #keep track that it's been called:
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

    def show_on(self, plot2, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        
        if output =='file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            #This must be the first time calling output_notebook,
            #keep track that it's been called:
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
                self.xdata)), 2)), 1/2) #analysis:ignore

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
                e.Measurement(self.pars_fit[i], self.pars_err[i], name=name),)
        self.flag['fitted'] = True

    def first_year_show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        
        if output =='file' or not qu.in_notebook():
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])
        elif not qu.bokeh_ouput_notebook_called:
            bi.output_notebook()
            #This must be the first time calling output_notebook,
            #keep track that it's been called:
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


def _error_bar(self, residual=False, xdata=None, ydata=None, plot_object=None,
               color=None):
    '''Function to create a Bokeh glyph which appears to be a datapoint with
    an errorbar.

    The datapoint is created using a Bokeh circle glyph. The errobars consist
    of two lines spanning error range of each datapoint. The errorbar caps
    are rectangles at each end of the errorline. The errorbar caps are
    rectangles whose size is based on Bokeh 'size' which is based on screen
    size and thus does not get larger when zooming in on a plot.
    '''

    if color is None:
        data_color = self.colors['Data Points'][0]
    elif type(color) is int:
        data_color = self.colors['Data Points'][color]
    else:
        print('Color option must be an integer')
        data_color = self.colors['Data Points'][0]

    if plot_object is None:
        if residual is False:
            p = self.p
        else:
            p = self.res
    else:
        if residual is False:
            p = plot_object.p
        else:
            p = plot_object.res

    if residual is True:
        data_name = None
    elif type(self.attributes['data']) is str:
        data_name = self.attributes['data']
    else:
        if plot_object is None:
            data_name = 'Data'
        else:
            data_name = 'Second Data'

    err_x1 = []
    err_d1 = []
    err_y1 = []
    err_d2 = []
    err_t1 = []
    err_t2 = []
    err_b1 = []
    err_b2 = []

    if xdata is None:
        _xdata = list(self.xdata)
        x_data = list(self.xdata)
    else:
        _xdata = [xdata.mean]
        x_data = [xdata.mean]

    if residual is True:
        _ydata = list(self.yres)
        y_res = list(self.yres)
        _yerr = list(self.yerr)

    elif ydata is None:
        _ydata = list(self.ydata)
        y_data = list(self.ydata)
        _yerr = list(self.yerr)
    else:
        _ydata = [ydata.mean]
        y_data = [ydata.mean]
        _yerr = [ydata.std]

    p.circle(_xdata, _ydata, color=data_color, size=2, legend=data_name)

    for _xdata, _ydata, _yerr in zip(_xdata, _ydata, _yerr):
        err_x1.append((_xdata, _xdata))
        err_d1.append((_ydata - _yerr, _ydata + _yerr))
        err_t1.append(_ydata+_yerr)
        err_b1.append(_ydata-_yerr)

    p.multi_line(err_x1, err_d1, color=data_color, legend=data_name)
    p.rect(
        x=x_data*2, y=err_t1+err_b1,
        height=0.2, width=5,
        height_units='screen', width_units='screen',
        color=data_color,
        legend=data_name)

    if xdata is None:
        _xdata = list(self.xdata)
        x_data = list(self.xdata)
        _xerr = list(self.xerr)
    else:
        _xdata = [xdata.mean]
        x_data = [xdata.mean]
        _xerr = [xdata.std]

    if residual is True:
        _ydata = list(self.yres)
        y_res = list(self.yres)
    elif ydata is None:
        _ydata = list(self.ydata)
        y_data = list(self.ydata)
    else:
        _ydata = [ydata.mean]
        y_data = [ydata.mean]

    for _ydata, _xdata, _xerr in zip(_ydata, _xdata, _xerr):
        err_y1.append((_ydata, _ydata))
        err_d2.append((_xdata - _xerr, _xdata + _xerr))
        err_t2.append(_xdata+_xerr)
        err_b2.append(_xdata-_xerr)

    p.multi_line(err_d2, err_y1, color=data_color, legend=data_name)
    if residual is True:
        p.circle(x_data, y_res, color=data_color, size=2)

        p.rect(
            x=err_t2+err_b2, y=y_res*2,
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=data_color)
    else:
        p.rect(
            x=err_t2+err_b2, y=y_data*2,
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=data_color, legend=data_name)


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


def data_transform(self, x, y, xerr=None, yerr=None):
    '''Function to interpret user inputted data into a form compatible with
    Plot objects.

    Based on various input cases, the user given data is transformed into two
    lists for x and y containing the data values and errors. These values and
    any values included in measurement objects, should those objects be
    inputted, are stored as object attributes. Information such as data units,
    and the name of datasets are stored.'''

    def _plot_arguments(arg, arg_err=None):
        '''Creates data and error for various inputs.
        '''
        if type(arg) is e.ExperimentalValue:
            # For input of Measurement object
            arg_measurement = e.MeasurementArray(arg.info['Data'],
                                                 arg.info['Error'],
                                                 name=arg.name)
            arg_data = arg_measurement.info['Data']
            arg_error = arg_measurement.info['Error']
            if arg_error is None:
                arg_error = [0]*len(arg_data)

        elif type(arg) is np.ndarray and\
                all(isinstance(n, e.ExperimentalValue) for n in arg):
            # For input of array of Measurement objects
            arg_data = []
            arg_error = []
            for val in arg:
                arg_data.append(val.mean)
                arg_error.append(val.std)
            arg_measurement = e.MeasurementArray(arg_data, arg_error)

        elif type(arg) in ARRAY and\
                all(isinstance(n, CONSTANT) for n in arg):
            # For input of array of values to be plotted
            arg_data = []
            for val in arg:
                arg_data.append(val)

            if arg_err is not None:
                arg_error = []
                for val in arg_err:
                    arg_error.append(val)
            else:
                arg_error = [0]*len(arg)
            arg_measurement = e.MeasurementArray(arg_data, arg_error)
        else:
            raise TypeError('Input method not recognized.')

        return (arg_data, arg_error, arg_measurement)

    def _plot_labels(arg):
        unit_string = arg[0].get_units()

        if type(arg) is np.ndarray and\
                all(isinstance(n, e.ExperimentalValue) for n in arg):
            arg_name = arg[0].name
        else:
            arg_name = arg[0].name

        return (unit_string, arg_name,)

    self.xdata, self.xerr, x = _plot_arguments(x, xerr)
    self.ydata, self.yerr, y = _plot_arguments(y, yerr)
    self.xunits, self.xname = _plot_labels(x)
    self.yunits, self.yname = _plot_labels(y)


def update_plot(self):
    ''' Creates interactive sliders in Jupyter Notebook to adjust fit.
    '''
    from ipywidgets import interact

    range_argument = ()
    for par in self.fit_parameters:
        min_val = par.mean - 2*par.std
        increment = (par.mean-min_val)/100
        range_argument += (min_val, par.mean, increment)

    for par in self.fit_parameters:
        increment = (par.std)/100
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
