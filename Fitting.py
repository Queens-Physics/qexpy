from scipy.optimize import curve_fit
import numpy as np
from uncertainties import Measured as M
from math import pi

from bokeh.plotting import figure, show
from bokeh.io import output_file, gridplot

ARRAY = (list, tuple, )


class Plot:
    '''
    Class of objects which can be plotted to display measurement objects
    which contain data and error data.
    '''

    def polynomial(x, pars):
        poly = 0
        n = 0
        for par in pars:
            poly += par*x**n
            n += 1
        return poly

    def gauss(x, pars):
        from numpy import exp
        mean, std = pars
        return (2*pi*std**2)**(-1/2)*exp(-(x-mean)**2/2/std**2)

    fits = {
            # 'linear': lambda x, pars: pars[0]+pars[1]*x,
            'linear': polynomial,
            'exponential': lambda x, pars: np.exp(pars[0]+pars[1]*x),
            'polynomial': polynomial,
            'gaussian': gauss}

    def mgauss(x, pars):
        from operations import exp
        mean, std = pars
        return (2*pi*std**2)**(-1/2)*exp(-(x-mean)**2/2/std**2)

    mfits = {
            # 'linear': lambda x, pars: pars[0]+pars[1]*x,
            'linear': polynomial,
            'exponential': lambda x, pars: np.exp(pars[0]+pars[1]*x),
            'polynomial': polynomial,
            'gaussian': mgauss}

    def __init__(self, x, y, xerr=None, yerr=None):
        '''
        Object which can be plotted.
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
            'title': x.name+' versus '+y.name, 'xaxis': 'x '+self.xunits,
            'yaxis': 'y '+self.yunits, 'data': 'Experiment', 'function': (), }
        self.fit_parameters = ()
        self.yres = None
        self.function_counter = 0
        self.manual_data = ()

    def residuals(self):

        if self.flag['fitted'] is False:
            Plot.fit(self.fit_function)

        # Calculate residual values
        yfit = self.fit_function(self.xdata, self.pars_fit)
        # Generate a set of residuals for the fit
        self.yres = self.ydata-yfit

        self.flag['residuals'] = True

    def fit(self, model=None, guess=None):

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

        self.pars_fit, self.pcov = curve_fit(
                                    model, self.xdata, self.ydata,
                                    sigma=self.yerr, p0=pars_guess)
        self.pars_err = np.sqrt(np.diag(self.pcov))
        for i in range(len(self.pars_fit)):
            if i is 0:
                name = 'intercept'
            elif i is 1:
                name = 'slope'
            else:
                name = 'parameter %d' % (i)

            self.fit_parameters += (M(self.pars_fit[i], self.pars_err[i],
                                      name=name),)
        self.flag['fitted'] = True

    def function(self, function):
        self.attributes['function'] += (function,)

    def show(self):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''

        output_file(self.plot_para['filename']+'.html',
                    title=self.attributes['title'])

        # create a new plot
        p = figure(
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
        error_bar(self, p)

        if self.flag['Manual'] is True:
            error_bar(self, p,
                      xdata=self.manual_data[0], ydata=self.manual_data[1])

        if self.flag['fitted'] is True:
            self.mfit_function = Plot.mfits[self.fit_method]
            _plot_function(
                self, p, self.xdata,
                lambda x: self.mfit_function(x, self.fit_parameters),
                color=self.colors['Function'][self.function_counter])

            self.function_counter += 1

            if self.fit_parameters[1].mean > 0:
                p.legend.orientation = "top_left"
            else:
                p.legend.orientation = "top_right"
        else:
            p.legend.orientation = 'top_right'

        for func in self.attributes['function']:
            _plot_function(
                self, p, self.xdata, func,
                color=self.colors['Function'][self.function_counter])
            self.function_counter += 1

        if self.flag['residuals'] is False:
            show(p)
        else:

            p2 = figure(
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
            error_bar(self, p2, residual=True)

            gp_alt = gridplot([[p], [p2]])
            show(gp_alt)

    def set_colors(self, data=None, error=None, line=None):
        if data is not None:
            self.colors['Data Points'] = data
        if error is not None:
            self.colors['Error'] = error
        if type(line) is str:
            self.colors['Function'][0] = line
        elif len(line) <= 3:
            for i in range(len(line)):
                self.colors['Function'][i] = line
        elif len(line) > 3 and type(line) in ARRAY:
            self.colors['Function'] = list(line)

    def set_name(self, title=None, xlabel=None, ylabel=None, data_name=None, ):
        if title is not None:
            self.attributes['title'] = title
        if xlabel is not None:
            self.attributes['xname'] = xlabel
        if ylabel is not None:
            self.attributes['yname'] = ylabel
        if data_name is not None:
            self.attributes['Data'] = data_name

    def manual_errorbar(self, data, function):
        from operations import check_values
        data, function = check_values(data, function)
        self.manual_data = (data, function(data))
        self.flag['Manual'] = True


def error_bar(self, p, residual=False, xdata=None, ydata=None):
    # create the coordinates for the errorbars
    err_x1 = []
    err_d1 = []
    err_y1 = []
    err_d2 = []
    err_t1 = []
    err_t2 = []
    err_b1 = []
    err_b2 = []

    if xdata is None:
        _xdata = self.xdata
        x_data = self.xdata
    else:
        _xdata = [xdata.mean]
        x_data = [xdata.mean]

    if residual is True:
        _ydata = self.yres
        y_res = self.yres
        _yerr = self.yerr

    elif ydata is None:
        _ydata = self.ydata
        y_data = self.ydata
        _yerr = self.yerr
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

    p.multi_line(err_x1, err_d1, color='red')
    p.rect(
        x=[*x_data, *x_data], y=[*err_t1, *err_b1],
        height=0.2, width=5,
        height_units='screen', width_units='screen', color='red')

    if xdata is None:
        _xdata = self.xdata
        x_data = self.xdata
        _xerr = self.xerr
    else:
        _xdata = [xdata.mean]
        x_data = [xdata.mean]
        _xerr = [xdata.std]

    if residual is True:
        _ydata = self.yres
        y_res = self.yres
    elif ydata is None:
        _ydata = self.ydata
        y_data = self.ydata
    else:
        _ydata = [ydata.mean]
        y_data = [ydata.mean]

    for _ydata, _xdata, _xerr in zip(_ydata, _xdata, _xerr):
        err_y1.append((_ydata, _ydata))
        err_d2.append((_xdata - _xerr, _xdata + _xerr))
        err_t2.append(_xdata+_xerr)
        err_b2.append(_xdata-_xerr)

    p.multi_line(err_d2, err_y1, color='red')
    if residual is True:
        p.circle(x_data, y_res, color=self.colors['Data Points'], size=2)

        p.rect(
            x=[*err_t2, *err_b2], y=[*y_res, *y_res],
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=self.colors['Data Points'])
    else:
        p.rect(
            x=[*err_t2, *err_b2], y=[*y_data, *y_data],
            height=5, width=0.2, height_units='screen', width_units='screen',
            color=self.colors['Data Points'])


def data_transform(self, x, y, xerr=None, yerr=None):
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
    self.xdata = xdata
    self.ydata = ydata
    self.xerr = x_error
    self.yerr = y_error
    self.xunits = xunits
    self.yunits = yunits


def _plot_function(self, p, xdata, theory, n=1000, color='red'):
    xrange = np.linspace(min(xdata), max(xdata), n)
    x_theory = theory(min(xdata))
    x_mid = []
    try:
        x_theory.type
    except AttributeError:
        for i in range(n):
            x_mid.append(theory(xrange[i]))
        p.line(
            xrange, x_mid, legend='Theoretical',
            line_color=color)
    else:
        x_max = []
        x_min = []
        for i in range(n):
            x_theory = theory(xrange[i])
            x_mid.append(x_theory.mean)
            x_max.append(x_theory.mean+x_theory.std)
            x_min.append(x_theory.mean-x_theory.std)
        p.line(
            xrange, x_mid, legend='Theoretical',
            line_color=color)

        xrange_reverse = list(reversed(xrange))
        x_min_reverse = list(reversed(x_min))
        p.patch(
            x=[*xrange, *xrange_reverse], y=[*x_max, *x_min_reverse],
            fill_alpha=0.3, fill_color=color,
            line_color=color, line_dash='dashed',
            line_alpha=0.3)
