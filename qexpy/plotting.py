import scipy.optimize as sp
import numpy as np
import qexpy.error as qe
import qexpy.utils as qu
import qexpy.fitting as qf
import qexpy.plot_utils as qpu

from math import pi
import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.models as mo
import bokeh.palettes as bpal

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

    def __init__(self, x=None, y=None, xerr=None, yerr=None, data_name=None, dataset=None):
        '''
        Constructor requiring two measeurement objects, or lists of data and
        error values.

        Plotting objects do not create Bokeh objects until shown or if the
        Bokeh object is otherwise requested. Class methods built here are
        used to record the data to be plotted and track what the user has
        requested to be plotted.
        '''
    
        self.color_palette = bpal.Set1_9
        self.color_count = 0
    
        #Revised members:
        self.datasets=[]
        if dataset != None:
            self.datasets.append(dataset)
        else:
            self.datasets.append(qf.XYDataSet(x, y, xerr=xerr, yerr=yerr, data_name=data_name))
        
        self.datasets_colors=[]
        self.datasets_colors.append(self._get_color_from_palette())
            
        self.xunits = self.datasets[-1].xunits
        self.xname = self.datasets[-1].xname
        
        self.yunits = self.datasets[-1].yunits
        self.yname = self.datasets[-1].yname    
        
        self.errorband_sigma = 1.0
        self.show_residuals=False
        

        
        self.x_range_margin = 0.5
        self.y_range_margin = 0.5
        self.x_range = self.datasets[-1].get_x_range(self.x_range_margin)
        self.y_range = self.datasets[-1].get_y_range(self.y_range_margin)
        
        self.dimensions = [600, 400]
        
        self.user_functions_count=0
        self.user_functions = []
        self.user_functions_pars = []
        self.user_functions_names = []
        self.user_functions_colors = []
        
        
        self.axes = {'xscale': 'linear', 'yscale': 'linear'}
        self.labels = {
            'title': self.datasets[-1].name,
            'xtitle': self.xname+' ['+self.xunits+']',
            'ytitle': self.yname+' ['+self.yunits+']',
            'data': data_name }
        
        
        
        self.plot_para = {
            'xscale': 'linear', 'yscale': 'linear', 'filename': 'Plot'}
        
        self.flag = {'fitted': False, 'residuals': False,
                     'Manual': False} # analysis:ignore
        self.attributes = {
            'title': self.datasets[-1].name,
            'xaxis': self.xname+' ['+self.xunits+']',
            'yaxis': self.yname+' ['+self.yunits+']',
            'data': data_name, 'function': (), }
        self.sigma = 1       

    def _get_color_from_palette(self):
        self.color_count += 1
        return self.color_palette[self.color_count-1]
    
    def fit(self, model=None, parguess=None, fit_range=None, datasetindex=-1):
        return self.datasets[datasetindex].fit()
        
    def print_fit_parameters(self, dataset=-1):
        if self.datasets[-1].nfits>0:
            print("Fit parameters:\n"+str(self.datasets[dataset].fit_pars[-1]))    
        else:
            print("Datasets have not been fit")
            
###############################################################################
# User Methods for adding to Plot Objects
###############################################################################

    def add_residuals(self):
        '''Add a subfigure with residuals to the main figure when plotting'''
        if self.datasets[-1].nfits>0:
            self.show_residuals = True

    
    def add_function(self, function, pars = None, name=None, color=None):
        '''Adds a specified function to the list of functions to be plotted.

        Functions are only plotted when a Bokeh object is created, thus user
        specified functions are stored to be plotted later.
        '''
        
        xvals = np.linspace(self.x_range[0],self.x_range[1], 100)
        
        #check if we should change the y-axis range to accomodate the function
        if not isinstance(pars, np.ndarray) and pars == None:
            fvals = function(xvals)
        elif isinstance(pars, qe.Measurement_Array) :
            recall = qe.Measurement.minmax_n
            qe.Measurement.minmax_n=1
            fmes = function(xvals, *(pars))
            fvals = fmes.get_means()
            qe.Measurement.minmax_n=recall
        elif isinstance(pars,(list, np.ndarray)):
            fvals = function(xvals, *pars)
        else:
            print("Error: Not a recognized format for parameter")
            return
                 
        fmax = fvals.max()+self.y_range_margin
        fmin = fvals.min()-self.y_range_margin
        
        if fmax > self.y_range[1]:
            self.y_range[1]=fmax
        if fmin < self.y_range[0]:
            self.y_range[0]=fmin
            
        self.user_functions.append(function)
        self.user_functions_pars.append(pars)
        fname = "userf_{}".format(self.user_functions_count) if name==None else name
        self.user_functions_names.append(fname)
        self.user_functions_count +=1
        
        if color is None:
            self.user_functions_colors.append(self._get_color_from_palette())
        else: 
            self.user_functions_colors.append(color)
        
    def add_dataset(self, dataset, color=None, name = None):
        self.datasets.append(dataset)
        if color is None:
            self.datasets_colors.append(self._get_color_from_palette())
        else: 
            self.datasets_colors.append(color)
        if name != None:
            self.datasets[-1].name=name
            
        x_range = dataset.get_x_range(self.x_range_margin)
        y_range = dataset.get_y_range(self.y_range_margin)
        
        if x_range[0] < self.x_range[0]:
            self.x_range[0]=x_range[0]
        if x_range[1] > self.x_range[1]:
            self.x_range[1]=x_range[1]
            
        if y_range[0] < self.y_range[0]:
            self.y_range[0]=y_range[0]
        if y_range[1] > self.y_range[1]:
            self.y_range[1]=y_range[1] 
 

###############################################################################
# Methods for changing parameters of Plot Object
###############################################################################

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

            
    def set_labels(self, title=None, xtitle=None, ytitle=None, data_name=None ):
        '''Change the labels for plot axis, datasets, or the plot itself.

        Method simply overwrites the automatically generated names used in
        the Bokeh plot.'''
        if title is not None:
            self.labels['title'] = title

        if xtitle is not None:
            self.labels['xtitle'] = xtitle

        if ytitle is not None:
            self.labels['ytitle'] = ytitle

        if data_name is not None:
            self.labels['Data'] = data_name

    def resize_plot(self, width=None, height=None):
        if width is None:
            width = 600
        if height is None:
            height = 400
        self.dimensions = [width, height]

    def set_errorband_sigma(self, sigma=1):
        '''Change the confidence bounds of the error range on a fit.
        '''
        self.errorband_sigma = sigma

###############################################################################
# Methods for Returning or Rendering Bokeh
###############################################################################
            
    def show(self, output='inline'):
        '''
        Method which creates and displays plot.
        Previous methods simply edit parameters which are used here, to
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

        bp.show(self.populate_bokeh_figure())
                    
            
    def populate_bokeh_figure(self):    
        # create a new bokeh figure
        self.bkfigure = self.initialize_bokeh_figure(residuals=False)
        
        # create the one for residuals if needed
        if self.show_residuals:
            self.bkres = self.initialize_bokeh_figure(residuals=True)
                              
        #plot the datasets and their latest fit
        count = 0
        for dataset in self.datasets:
            color = self.datasets_colors[count]
            qpu.plot_dataset(self.bkfigure, dataset, residual=False,
                            data_color=color)
                   
            if dataset.nfits>0:
                qpu.plot_function(self.bkfigure, function=dataset.fit_function[-1],
                                  xdata=dataset.xdata,fpars=dataset.fit_pars[-1],
                                  n=100, legend_name=dataset.fit_function_name[-1],
                                  color=color, errorbandfactor=self.errorband_sigma)
                
                #Draw fit parameters only for the first 2 dataset
                if count<1:
                     for i in range(dataset.fit_npars[-1]):
                        #shorten the name of the fit parameters
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
                        self.bkfigure.add_layout(citation)
                        
                if self.show_residuals:
                    qpu.plot_dataset(self.bkres, dataset, residual=True,
                                     data_color=color)
            count += 1

        #Now add any user defined functions:
        xvals = np.linspace(self.x_range[0]+self.x_range_margin,self.x_range[1]-self.x_range_margin, 100)
   
        count = 0
        for func, pars, fname in zip(self.user_functions,self.user_functions_pars,self.user_functions_names):
            color = self.user_functions_colors[count]
            qpu.plot_function(self.bkfigure, function=func, xdata=xvals,
                              fpars=pars, n=100, legend_name= fname,
                              color=color, errorbandfactor=self.errorband_sigma)
            count += 1
        
        #Specify the location of the legend
        self.bkfigure.legend.location = "top_left"
        
        if self.flag['residuals']:
            self.bkfigure = bi.gridplot([[self.bkfigure], [self.bkres]])
            
        return self.bkfigure
    
    def initialize_bokeh_figure(self, residuals=False):  
        if residuals==False:
            return bp.figure(
                tools='save, pan, box_zoom, wheel_zoom, reset',
                width=self.dimensions[0], height=self.dimensions[1],
                y_axis_type=self.axes['yscale'],
                y_range=self.y_range,
                x_axis_type=self.axes['xscale'],
                x_range=self.x_range,
                title=self.attributes['title'],
                x_axis_label=self.attributes['xaxis'],
                y_axis_label=self.attributes['yaxis'],
            )
        else:
            return bp.figure(
                width=self.dimensions[0], height=self.dimensions[1]//3,
                tools='save, pan, box_zoom, wheel_zoom, reset',
                y_axis_type='linear',
                y_range=self.datasets[-1].get_yres_range(),
                x_range=self.bkfigure.x_range,
                x_axis_label=self.attributes['xaxis'],
                y_axis_label='Residuals'
            )
        
    def show_interactive_linear_fit():
        pass
   
    
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
