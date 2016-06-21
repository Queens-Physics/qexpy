import scipy.optimize as sp
import numpy as np
import error as e
from math import pi
import bokeh.plotting as bp
import bokeh.io as bi

ARRAY = (list, tuple, )


class Plot:
    '''Objects which contain a dataset and any number of fuctions which can
    be shown on a Bokeh plot

    Plot objects are indirectly dependent on Bokeh plots. Plot objects
    contain data and associated error lists, as well as functions or fits
    of the data. These points are used to contruct Bokeh plots by using
    custom functions to draw a series of circles, lines, and rectangles
    which mimic errorbars and data points.
    '''

    def polynomial(x, pars):
        '''Function for a polynomial of nth order, requiring n pars.'''
        poly = 0
        n = 0
        for par in pars:
            poly += np.multiply(par, np.power(x, n))
            n += 1
        return poly

    def gauss(x, pars):
        '''Fucntion of gaussian distribution'''
        from numpy import exp
        mean, std = pars
        return (2*pi*std**2)**(-1/2)*exp(-(x-mean)**2/2/std**2)

    fits = {
            'linear': lambda x, pars: pars[0]+np.multiply(pars[1], x),
            'exponential': lambda x, pars: np.exp(pars[0]+pars[1]*x),
            'polynomial': polynomial,
            'gaussian': gauss}

    def mgauss(x, pars):
        '''Altered gaussian function to handle measurement objects.'''
        from error_operations import exp
        mean, std = pars
        return (2*pi*std**2)**(-1/2)*exp(-(x-mean)**2/2/std**2)

    mfits = {
            # 'linear': lambda x, pars: pars[0]+pars[1]*x,
            'linear': lambda x, pars: pars[0]+np.multiply(pars[1], x),
            'exponential': lambda x, pars: np.exp(pars[0]+pars[1]*x),
            'polynomial': polynomial,
            'gaussian': mgauss}

    def __init__(self, x, y, xerr=None, yerr=None):
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
            'Data Points': 'red', 'Function': ['blue', 'green', 'orange'],
            'Error': 'red'}
        self.fit_method = 'linear'
        self.fit_function = Plot.fits[self.fit_method] # analysis:ignore
        self.mfit_function = Plot.mfits[self.fit_method]
        self.plot_para = {
            'xscale': 'linear', 'yscale': 'linear', 'filename': 'Plot'}
        self.flag = {'fitted': False, 'residuals': False,
                     'Manual': False} # analysis:ignore
        self.attributes = {
            'title': self.xname+' versus '+self.yname,
            'xaxis': 'x '+self.xunits, 'yaxis': 'y '+self.yunits,
            'data': 'Experiment', 'function': (), }
        self.fit_parameters = ()
        self.yres = None
        self.function_counter = 0
        self.manual_data = ()

    def residuals(self):
        '''Request residual output for plot.'''
        if self.flag['fitted'] is False:
            Plot.fit(self.fit_function)

        # Calculate residual values
        yfit = self.fit_function(self.xdata, self.pars_fit)
        # Generate a set of residuals for the fit
        self.yres = self.ydata-yfit

        self.flag['residuals'] = True

    def fit(self, model=None, guess=None):
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
        if guess is None:
            if model == 'linear':
                guess = [1, 1]
            elif model[0] is 'p':
                degree = int(model[len(model)-1]) + 1
                guess = [1]*degree
            elif model is 'gaussian':
                guess = [1]*2

        if model is not None:
            if type(model) is not str:
                self.fit_function = model
                self.flag['Unknown Function'] = True
            elif model[0] is 'p' and model[1] is 'o':
                model = 'polynomial'
            else:
                self.fit_method = model

        self.fit_function = Plot.fits[self.fit_method]

        def model(x, *pars):
            return self.fit_function(x, pars)

        pars_guess = guess

        self.pars_fit, self.pcov = sp.curve_fit(
                                    model, self.xdata, self.ydata,
                                    sigma=self.yerr, p0=pars_guess)
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
                name = 'parameter %d' % (i)

            self.fit_parameters += (
                e.Measurement(self.pars_fit[i], self.pars_err[i], name=name),)
        self.flag['fitted'] = True

    def function(self, function):
        '''Adds a specified function to the list of functions to be plotted.

        Functions are only plotted when a Bokeh object is created, thus user
        specified functions are stored to be plotted later.
        '''
        self.attributes['function'] += (function,)

    def show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        if output is 'inline':
            bi.output_notebook()
        elif output is 'file':
            bi.output_file(self.plot_para['filename']+'.html',
                           title=self.attributes['title'])

        # create a new plot
        self.p = bp.figure(
            tools="pan, box_zoom, reset, save, wheel_zoom",
            width=600, height=400,
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
            self.mfit_function = Plot.mfits[self.fit_method]
            _plot_function(
                self, self.xdata,
                lambda x: self.mfit_function(x, self.fit_parameters))

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
        else:

            self.p2 = bp.figure(
                tools="pan, box_zoom, reset, save, wheel_zoom",
                width=600, height=200,
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=[min(self.xdata)-2*max(self.xerr),
                         max(self.xdata)+2*max(self.xerr)],
                title="Residual Plot",
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(self, residual=True)

            gp_alt = bi.gridplot([[self.p], [self.p2]])
            bp.show(gp_alt)

    def set_colors(self, data=None, error=None, line=None):
        '''Method to changes colors of data or function lines.

        User can specify a list or tuple of color strings for the data points
        or functions, as multiple datasets and functions will be plotted with
        different functions.
        '''
        if data is not None:
            self.colors['Data Points'] = data
        if error is not None:
            self.colors['Error'] = error
        if type(line) is str:
            self.colors['Function'][0] = line
            if len(line) <= 3:
                for i in range(len(line)):
                    self.colors['Function'][i] = line
            elif len(line) > 3 and type(line) in ARRAY:
                self.colors['Function'] = list(line)

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

    def manual_errorbar(self, data, function):
        '''Manually specify the location of a datapoint with errorbars.'''
        from error_operations import check_values
        data, function = check_values(data, function)
        self.manual_data = (data, function(data))
        self.flag['Manual'] = True

    def return_bokeh(self):
        '''Return Bokeh plot object for the plot acted upon.

        If no residual plot exists, a single Bokeh object, containing the
        main plot is returned. Else, a tuple with the main and residual plot
        in that order is returned.'''
        # create a new plot
        self.p = bp.figure(
            tools="pan, box_zoom, reset, save, wheel_zoom",
            width=600, height=400,
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
            self.mfit_function = Plot.mfits[self.fit_method]
            _plot_function(
                self, self.xdata,
                lambda x: self.mfit_function(x, self.fit_parameters))

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
            self.p2 = bp.figure(
                tools="pan, box_zoom, reset, save, wheel_zoom",
                width=600, height=200,
                y_axis_type='linear',
                y_range=[min(self.yres)-2*max(self.yerr),
                         max(self.yres)+2*max(self.yerr)],
                x_range=[min(self.xdata)-2*max(self.xerr),
                         max(self.xdata)+2*max(self.xerr)],
                title="Residual Plot",
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )

            # plot y errorbars
            _error_bar(self, residual=True)
            return (self.p, self.p2)


def _error_bar(self, residual=False, xdata=None, ydata=None):
    '''Function to create a Bokeh glyph which appears to be a datapoint with
    an errorbar.

    The datapoint is created using a Bokeh circle glyph. The errobars consist
    of two lines spanning error range of each datapoint. The errorbar caps
    are rectangles at each end of the errorline. The errorbar caps are
    rectangles whose size is based on Bokeh 'size' which is based on screen
    size and thus does not get larger when zooming in on a plot.
    '''
    if residual is False:
        p = self.p
    else:
        p = self.p2

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

    p.circle(_xdata, _ydata, color=self.colors['Data Points'], size=2)

    for _xdata, _ydata, _yerr in zip(_xdata, _ydata, _yerr):
        err_x1.append((_xdata, _xdata))
        err_d1.append((_ydata - _yerr, _ydata + _yerr))
        err_t1.append(_ydata+_yerr)
        err_b1.append(_ydata-_yerr)

    p.multi_line(err_x1, err_d1, color=self.colors['Data Points'])
    p.rect(
        x=x_data*2, y=err_t1+err_b1,
        height=0.2, width=5,
        height_units='screen', width_units='screen',
        color=self.colors['Data Points'])

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

    p.multi_line(err_d2, err_y1, color=self.colors['Data Points'])
    if residual is True:
        p.circle(x_data, y_res, color=self.colors['Data Points'], size=2)

        p.rect(
            x=err_t2+err_b2, y=y_res*2,
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=self.colors['Data Points'])
    else:
        p.rect(
            x=err_t2+err_b2, y=y_data*2,
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=self.colors['Data Points'])


def show_bokeh(self, p, p2=None):
    self.p = p
    if p2 is None:
        bp.show(p)
    else:
        self.p2 = p2
        gp_alt = bi.gridplot([[p], [p2]])
        bp.show(gp_alt)


def data_transform(self, x, y, xerr=None, yerr=None):
    '''Function to interpret user inputted data into a form compatible with
    Plot objects.

    Based on various input cases, the user given data is transformed into two
    lists for x and y containing the data values and errors. These values and
    any values included in measurement objects, should those objects be
    inputted, are stored as object attributes. Information such as data units,
    and the name of datasets are stored.'''
    if xerr is None:
        xdata = x.info['Data']
        x_error = x.info['Error']
    else:
        try:
            x.type
        except AttributeError:
            xdata = x
        else:
            xdata = x.info['Data']
        if type(xerr) in (int, float, ):
            x_error = [xerr]*len(xdata)
        else:
            x_error = xerr

    if yerr is None:
        ydata = y.info['Data']
        y_error = y.info['Error']
    else:
        try:
            y.type
        except AttributeError:
            ydata = y
        else:
            ydata = y.info['Data']
        if type(yerr) in (int, float, ):
            y_error = [yerr]*len(ydata)
        else:
            y_error = yerr

    try:
        x.units
    except AttributeError:
        xunits = ''
    else:
        if len(x.units) is not 0:
            xunits = ''
            for key in x.units:
                xunits += key+'^%d' % (x.units[key])
            xunits = '['+xunits+']'
        else:
            xunits = ''
    try:
        y.units
    except AttributeError:
        yunits = ''
    else:
        if len(y.units) is not 0:
            yunits = ''
            for key in y.units:
                yunits += key+'^%d' % (y.units[key])
            yunits = '['+yunits+']'
        else:
            yunits = ''

    try:
        x.name
    except AttributeError:
        xname = 'x'
    else:
        xname = x.name
    try:
        y.name
    except AttributeError:
        yname = 'y'
    else:
        yname = y.name

    self.xdata = xdata
    self.ydata = ydata
    self.xerr = x_error
    self.yerr = y_error

    self.xunits = xunits
    self.yunits = yunits
    self.xname = xname
    self.yname = yname


def _plot_function(self, xdata, theory, n=1000):
    '''Semi-privite function to plot a function over a given range of values.

    Curves are generated by creating a series of lines between points, the
    parameter n is the number of points. Line color is given by the plot
    attribute containing a list of colors which are assigned uniquly to a
    curve.'''
    xrange = np.linspace(min(xdata), max(xdata), n)
    x_theory = theory(min(xdata))
    x_mid = []
    try:
        x_theory.type
    except AttributeError:
        for i in range(n):
            x_mid.append(theory(xrange[i]))
        self.p.line(
            xrange, x_mid, legend='Theoretical',
            line_color=self.colors['Function'][self.function_counter])
    else:
        x_max = []
        x_min = []
        for i in range(n):
            x_theory = theory(xrange[i])
            x_mid.append(x_theory.mean)
            x_max.append(x_theory.mean+x_theory.std)
            x_min.append(x_theory.mean-x_theory.std)
        self.p.line(
            xrange, x_mid, legend='Theoretical',
            line_color=self.colors['Function'][self.function_counter])

        xrange_reverse = list(reversed(xrange))
        x_min_reverse = list(reversed(x_min))
        xrange = list(xrange)
        x_max = list(x_max)

        self.p.patch(
            x=xrange+xrange_reverse, y=[*x_max, *x_min_reverse],
            fill_alpha=0.3,
            fill_color=self.colors['Function'][self.function_counter],
            line_color=self.colors['Function'][self.function_counter],
            line_dash='dashed', line_alpha=0.3)
