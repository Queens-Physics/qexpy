import numpy as np
#used for the histograms, will remove when move bokeh histo to 
#to use Plot():
import bokeh.plotting as bp
import bokeh.io as bi
#used for array and number types:
import qexpy.utils as qu
#used to check plot_engine:
import qexpy as q


class ExperimentalValue:
    '''Root class of objects which containt a mean and standard deviation.
    From this class, objects with properties pertaining to their use or
    formulation can be instanced. (ie. the result of an operation of
    measured values, called Funciton and Measured respectivly)
    '''
    _error_method = "Derivative"  # Default error propogation method
    print_style = "Default"  # Default printing style
    mc_trial_number = 10000  # number of trial in Monte Carlo simulation
    minmax_n = 100 # grid size in MinMax calculation

    figs = None
    figs_on_uncertainty = False
    register = {}
    formula_register = {}
    id_register = {}

    # Defining common types under single arrayclear
    from numpy import int64, float64, ndarray, int32, float32
    CONSTANT = qu.number_types #(int, float, int64, float64, int32, float32)
    ARRAY = qu.array_types #(list, tuple, ndarray)

    def __init__(self, *args, name=None):
        '''Creates a variable that contains a mean, standard deviation,
        and name for inputted data.
        '''
        self.der = [0, 0]
        self.MinMax = [0, 0]
        self.MC = [0, 0]

        self.info = {
                'ID': '', 'Formula': '', 'Method': '', 'Data': [],\
                'Function': {
                        'operation': (), 'variables': ()}, }

        if len(args) ==1:
            if isinstance(args[0], qu.array_types):
                data = np.ndarray(len(args[0]))
                self.info['Data'] = data
                for index in range(len(args[0])):
                    data[index] = args[0][index]
                self.mean = data.mean()
                self.std = data.std(ddof=1)
                self.error_on_mean = 0 if data.size==0 else self.std/np.sqrt(data.size)             
            else:
                raise TypeError('''Input must be either a single array of values,
                      or the central value and uncertainty in one measurement''')
        elif len(args)==2:
            if isinstance(args[0], qu.number_types) and isinstance(args[1], qu.number_types):
                self.mean = float(args[0])
                data = np.ndarray(1)
                error_data = np.ndarray(1)
                data[0] = self.mean
                self.info['Data'] = data
                self.std = float(args[1])
            elif isinstance(args[0], qu.array_types) and isinstance(args[1], qu.array_types):
                raise TypeError('''Input must be either a single array of values,
                      or the central value and uncertainty in one measurement.
                      
                      The feature of passing a list of measurements and their corresponding
                      uncertanties is now deprecated. Please use a MeasurementArray insted.
                      More info: 
                      https://github.com/Queens-Physics/qexpy/blob/master/examples/jupyter/1_Intro_to_Error_Propagation.ipynb''')
            else:
                raise TypeError('''Input must be either a single array of values,
                      or the central value and uncertainty in one measurement''')
        else:
            raise TypeError('''Input must be either a single array of values,
                  or the central value and uncertainty in one measurement''')

        ExperimentalValue.id_register[id(self)] = self
        self.print_style = ExperimentalValue.print_style
        
        if name is not None:
            self.user_name = True
        else:
            self.user_name = False

        self.units = {}
        self.MC_list = None
        
###############################################################################
# Methods for printing or returning Measurement paramters
###############################################################################

    def __str__(self):
        '''Method called when printing measurement objects.'''
        if self.user_name:
            string = self.name+' = '
        else:
            string = ''

        if ExperimentalValue.print_style == "Latex":
            string += _tex_print(self)
        elif ExperimentalValue.print_style == "Default":
            string += _def_print(self)
        elif ExperimentalValue.print_style == "Scientific":
            string += _sci_print(self)

        unit_string = self.get_units()
        if unit_string != '':
            if ExperimentalValue.print_style == "Latex":
                string += '\,'+unit_string
            else:
                string += ' ['+unit_string+']'

        return string

    def print_mc_error(self):
        '''Prints the result of a Monte Carlo error propagation.

        The purpose of this method is to easily compare the results of a
        Monte Carlo propagation with whatever method is chosen.
        '''
        if self.print_style == "Latex":
            string = _tex_print(self, method="Monte Carlo")
        elif self.print_style == "Default":
            string = _def_print(self, method="Monte Carlo")
        elif self.print_style == "Scientific":
            string = _sci_print(self, method="Monte Carlo")
        print(string)

    def print_min_max_error(self):
        '''Prints the result of a Min-Max method error propagation.

        The purpose of this method is to easily compare the results of a
        Min-Max propagation with whatever method is chosen to confirm that
        the Min-Max is the upper bound of the error.
        '''
        if self.print_style == "Latex":
            string = _tex_print(self, method="Min Max")
        elif self.print_style == "Default":
            string = _def_print(self, method="Min Max")
        elif self.print_style == "Scientific":
            string = _sci_print(self, method="Min Max")
        print(string)

    def print_deriv_error(self):
        '''Prints the result of a Min-Max method error propagation.

        The purpose of this method is to easily compare the results of a
        Min-Max propagation with whatever method is chosen to confirm that
        the Min-Max is the upper bound of the error.
        '''
        if self.print_style == "Latex":
            string = _tex_print(self, method="Derivative")
        elif self.print_style == "Default":
            string = _def_print(self, method="Derivative")
        elif self.print_style == "Scientific":
            string = _sci_print(self, method="Derivative")
        print(string)

    def get_derivative(self, variable=None):
        '''Returns the numerical value of the derivative with respect to the
        inputed variable.

        Function to find the derivative of a measurement or measurement like
        object. By default, derivative is with respect to itself, which will
        always yeild 1. Operator acts by acessing the self.derivative
        dictionary and returning the value stored there under the specific
        variable inputted (ie. deriving with respect to variable = ???)
        '''
        if variable is not None \
                and type(variable) is not Measurement:
            print('''The derivative of a Measurement with respect to anything
                  other than a Measurement is zero.''')
            return 0

        elif variable is None:
            raise TypeError('''The object must be differentiated with respect to another
            Measurement.''')

        if variable.info['ID'] not in self.derivative:
            self.derivative[variable.info['ID']] = 0

        derivative = self.derivative[variable.info["ID"]]
        return derivative

    @property
    def mean(self):
        '''Returns the mean of a Measurement object.
        '''
        return self._mean

    @mean.setter
    def mean(self, mean):
        '''Sets the mean of a Measurement object.
        '''
        if(type(mean) in ExperimentalValue.CONSTANT):
            self.der[0] = mean
            self.MinMax[0] = mean
            self.MC[0] = mean
            self._mean = mean
        else:
            print("Mean must be a number")
            self._mean = 0

    @property
    def std(self):
        '''Returns the standard deviation of a Measurement object.
        '''
        return self._std

    @std.setter
    def std(self, std):
        '''Sets the standard deviation of a Measurement object.
        '''
        if(type(std) in ExperimentalValue.CONSTANT):
            self.der[1] = std
            self.MinMax[1] = std
            self.MC[1] = std
            self._error_on_mean = std/np.sqrt(len(self.get_data_array()))
            self._std = std
        else:
            print("Standard deviation must be a number")
            self._std = 0

    @property
    def error_on_mean(self):
        '''Returns the error on the mean of a Measurement object.
        '''
        if self._error_on_mean:
            return self._error_on_mean
        else:
            print("Error: error on mean not calculated")
            return 0

    @error_on_mean.setter
    def error_on_mean(self, error_on_mean):
        '''Sets the error on the mean of a Measurement object.
        '''
        if(type(error_on_mean) in ExperimentalValue.CONSTANT):
            self._error_on_mean = error_on_mean
            self._std = error_on_mean*np.sqrt(len(self.get_data_array()))
        else:
            print("Error on mean must be a number")
            self._error_on_mean = 0

    @property
    def name(self):
        '''Returns the name of a Measurement object.
        '''
        return self._name

    @name.setter
    def name(self, name):
        '''Sets the name of a Measurement object.
        '''
        if isinstance(name, str):
            self._name = name
        else:
            print("Name must be a string")
            self._name = 'unnamed_var%d' % (Measurement.id_number)

    @property
    def relative_error(self):
        '''Returns the relative error (error/mean) of a Measurement object.
        '''
        return self.std/self.mean if self.mean !=0 else 0. 

    @relative_error.setter
    def relative_error(self, rel_error):
        '''Sets the relative error (error/mean) of a Measurement object.
        '''
        if(type(rel_error) in qu.number_types):
            self._std = self.mean*rel_error
        else:
            print("Relative error must be a number")

    @property
    def error_method(self):
        '''Returns the method (Monte Carlo, derivative or min max) 
        used to calculate error of a Measurement object.
        '''
        return self._error_method

    @error_method.setter
    def error_method(self, method):
        '''Sets the method (Monte Carlo, derivative or min max) 
        used to calculate error of a Measurement object.
        '''
        mc_list = ('MC', 'mc', 'montecarlo', 'Monte Carlo', 'MonteCarlo',
                   'monte carlo',)
        min_max_list = ('Min Max', 'MinMax', 'minmax', 'min max',)
        derr_list = ('Derivative', 'derivative', 'diff', 'der', 'Default',
                     'default',)

        if method in mc_list:
            self._error_method = "Monte Carlo"
        elif method in min_max_list:
            self._error_method = "Min Max"
        elif method in derr_list:
            self._error_method = "Derivative"
        else:
            print("Method not recognized, using derivative method.")
            self._error_method = "Derivative"

    def get_data_array(self):
        '''Returns the underlying data array used to create the Measurement object.
        '''
        if self.info['Data'] is None:
            print('No data array exists.')
            return None
        return self.info['Data']

    def get_units(self):
        '''Returns the units of the associated Measurement.
        '''
        unit_string = ''
        if self.units != {}:
            for key in self.units:
                if self.units[key] == 1 and len(self.units.keys()) is 1:
                    unit_string = key + unit_string
                else:
                    unit_string += key+'^%d' % (self.units[key])
                    unit_string += ' '

                if unit_string == '':
                    unit_string = 'unitless'
        return unit_string
    
    def show_histogram(self, bins=50, color="#036564", title=None, output='inline'):
        '''Creates a histogram of the inputted data using Bokeh or mpl.
        '''
        if self.info['Data'] is None:
            print("no data to histogram")
            return None

        if type(title) is str:
            hist_title = title
        elif title is None:
            hist_title = self.name+' Histogram'
        else:
            print('Histogram title must be a string.')
            hist_title = self.name+' Histogram'                     
        
        data_arr = self.info['Data']

        data = q.XYDataSet(xdata = data_arr, is_histogram = True, data_name=hist_title)
        fig = q.MakePlot()
        fig.add_dataset(data, color=color)
        fig.x_range = [min(data_arr)*.95,max(data_arr)*1.05]
        fig.y_range = [0,max(data.ydata)*1.2]

        mean = self.mean
        std = self.std
        fig.add_line(x=mean, dashed=False, color='red')
        fig.add_line(x=mean+std, dashed=True, color='red')
        fig.add_line(x=mean-std, dashed=True, color='red')

        fig.show()
        return fig
    
    def show_MC_histogram(self, bins=50, color="#036564", title=None, output='inline'):
        '''Creates and shows a Bokeh plot of a histogram of the values
        calculated by Monte Carlo error simulation.
        '''
        MC_data = self.MC_list
        if MC_data is None:
            print("no MC data to histogram")
            return None

        if type(title) is str:
            hist_title = title
        elif title is None:
            hist_title = self.name+' Histogram'
        else:
            print('Histogram title must be a string.')
            hist_title = self.name+' Histogram'

        data = q.XYDataSet(xdata = MC_data, is_histogram = True, data_name=hist_title)
        fig = q.MakePlot()
        fig.add_dataset(data, color = color)
        fig.x_range = [min(MC_data)*.95,max(MC_data)*1.05]
        fig.y_range = [0,max(data.ydata)*1.2]

        # Adds a line at the mean and at the range corresponding to 68% coverage.
        MC_mean, MC_std = self.MC
        fig.add_line(x=MC_mean, dashed=False, color='red')
        fig.add_line(x=MC_mean+MC_std, dashed=True, color='red')
        fig.add_line(x=MC_mean-MC_std, dashed=True, color='red')

        fig.show()
        return fig

    def show_error_contribution(self, title=None, output='inline'):
        '''Creates and shows a Bokeh or mpl plot of a histogram of the relative
        contribution of individual measurements to the variance of a calculated value.
        '''
        terms = {}
        formula = self.info['Formula']

        # For each measurement that goes into the calculation, evaluate the calculation
        # at that measurement +/- the std. Take the output of that and do .5*(output-mean)^2.
        # Add the +std and -std term. 
        # This process is described in this paper: http://pubs.acs.org/doi/abs/10.1021/ed1004307 
        for i in self.root:
            maxx = formula
            minn = formula
            name = ""
            for j in self.root:
                meas = q.error.ExperimentalValue.register[j]
                if j == i:
                    name = meas.name
                    maxx = maxx.replace(j, str(meas.mean+meas.std))
                    minn = minn.replace(j, str(meas.mean-meas.std))
                else:
                    maxx = maxx.replace(j, str(meas.mean))
                    minn = minn.replace(j, str(meas.mean))
            terms[name] = .5*(eval(maxx)-self.mean)**2+.5*(eval(minn)-self.mean)**2
            
        N = len(terms)
        names = []
        vals = []

        for k, v in sorted(terms.items()):
            names.append(k)
            vals.append(v)

        # Change the absolute terms into relative terms.
        summ = sum(vals)
        for index in range(N):
            vals[index] = vals[index]/summ

        # Add spacing to make the histogram look like a bar chart.
        new_vals = []
        new_names = []
        for index in range(N):
            new_vals.append(vals[index])
            new_vals.append(0)
            new_names.append('')
            new_names.append(names[index])
        new_vals = new_vals[0:-1]

        data = q.XYDataSet(xdata=np.arange(2*N-1), ydata=new_vals, is_histogram = True, bins=N,
                    data_name='Relative contribution to variance of {}'.format(self.name))

        # Populates the mpl figure in case it is plotted.
        fig = q.MakePlot()
        fig.add_dataset(data, color='blue')
        fig.x_range = [-1,2*N-1]
        fig.y_range = [0,1]
        fig.set_labels(xtitle="", ytitle="")
        fig.populate_mpl_figure()
        fig.mplfigure_main_ax.axes.set_xticklabels(new_names)
        fig.mplfigure_main_ax.axes.grid(False, which='both', axis='x')

        # Populates the boken figure in case it is plotted.
        # The messy stuff comes from the fact that mpl boxes, 
        # mpl labels and bokeh boxes are 0-indexed, but mpl labels are 1 indexed.
        fig.axes['xscale'] = 'auto'
        fig.datasets[0].ydata = np.insert(fig.datasets[0].ydata, [0, 0, 2*N-1], [0, 0, 0])
        fig.datasets[0].xdata = np.append(fig.datasets[0].xdata, [2*N-1, 2*N, 2*N+1])
        fig.x_range = new_names+['']
        fig.populate_bokeh_figure()
        fig.bkfigure.xgrid.grid_line_color = None
        fig.bkfigure.xaxis.major_tick_line_color = None
        fig.bkfigure.xaxis.minor_tick_line_color = None

        # Will use whatever plotting engine is in use.
        fig.show(populate_figure=False, refresh = True)

###############################################################################
# Methods for Correlation and Covariance
###############################################################################

    def _find_covariance(x, y):
        '''Uses the data from which x and y were generated to calculate
        covariance and add this informaiton to x and y.

        Requires data arrays to be stored in the .info of both objects
        and that these arrays are of the same length, as the covariance is
        only defined in these cases.
        '''
        try:
            x.covariance[y.info['ID']]
            return
        except KeyError:
            pass

        data_x = x.info["Data"]
        data_y = y.info["Data"]

        if data_x is None or data_y is None:
            raise TypeError(
                "Data arrays must exist for both quantities " +
                "to define covariance.")

        if len(data_x) != len(data_y):
            raise TypeError('Lengths of data arrays must be equal to\
                      define a covariance')
            
        sigma_xy = np.sum((data_x-x.mean)*(data_y-y.mean))  
        nmin1 = len(data_x)-1
        if nmin1 != 0:
            sigma_xy /= nmin1
                    
        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

        factor = sigma_xy
        if x.std != 0:
            factor /= x.std
        if y.std != 0:
            factor /= y.std
              
        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

    def set_correlation(self, y, factor):
        '''Manually set the correlation between two quantities

        Given a correlation factor, the covariance and correlation
        between two variables is added to both objects.
        '''
        if factor > 1 or factor < -1:
            raise ValueError('Correlation factor must be between -1 and 1.')

        x = self
        rho_xy = factor
        sigma_xy = rho_xy*x.std*y.std

        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

    def set_covariance(self, y, sigma_xy):
        '''Manually set the covariance between two quantities

        Given a covariance value, the covariance and correlation
        between two variables is added to both objects.
        '''
        x = self
        factor = sigma_xy
        if x.std != 0:
            factor /= x.std
        if y.std != 0:
            factor /= y.std

        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

    def get_covariance(self, variable):
        '''Returns the covariance of the object and a specified variable.

        This funciton checks for the existance of a data array in each
        object and that the covariance of the two objects is not already
        specified. In each case, the covariance is returned, unless
        the data arrays are of different lengths or do not exist, in that
        case a covariance of zero is returned.
        '''
        if self.info['ID'] in variable.covariance:
            return self.covariance[variable.info['ID']]

        elif self.info["Data"] is not None \
                and variable.info["Data"] is not None\
                and len(self) == len(variable):
            ExperimentalValue._find_covariance(self, variable)
            var = self.covariance[variable.info['ID']]
            return var

        else:
            return 0

    def _get_correlation(self, y):
        '''Returns the correlation factor of two measurements.

        Using the covariance, or finding the covariance if not defined,
        the correlation factor of two measurements is returned.
        '''
        x = self

        if y.info['ID'] in x.covariance:
            pass
        else:
            # ExperimentalValue._find_covariance(x, y)
            return 0

        sigma_xy = x.covariance[y.info['ID']]
        sigma_x = x.std
        sigma_y = y.std
        
        factor = sigma_xy
        if x.std != 0:
            factor /= x.std
        if y.std != 0:
            factor /= y.std
            
        return factor

###############################################################################
# Methods for Naming and Units
###############################################################################

    def rename(self, newName=None, units=None):
        '''Renames an object, requires a string.
        '''
        if newName is not None:
            self.name = newName
            self.user_name = True

        if units is not None:
            if type(units) is str:
                self.units[units] = 1
            else:
                for i in range(len(units)//2):
                    self.units[units[2*i]] = units[i+1]

    def _update_info(self, operation, *args, func_flag=False):
        '''
        Update the formula, name and method of an object.

        Function to update the name, formula and method of a value created
        by a measurement operation. The name is updated by combining the
        names of the object acted on with another name using whatever
        operation is being performed. The function flag is to change syntax
        for functions like sine and cosine. Method is updated by acessing
        the class property.
        '''
        import qexpy.error_operations as op

        if len(args) is 1:
            var1 = args[0]
            var2 = None
        elif len(args) is 2:
            var1 = args[0]
            var2 = args[1]

        op_string = {op.sin: 'sin', op.cos: 'cos', op.tan: 'tan', op.sqrt: 'sqrt',
                     op.csc: 'csc', op.sec: 'sec', op.cot: 'cot',
                     op.exp: 'exp', op.log: 'log', op.add: '+',
                     op.sub: '-', op.mul: '*', op.div: '/', op.power: '**',
                     'neg': '-', op.asin: 'asin', op.acos: 'acos',
                     op.atan: 'atan', }

        if func_flag == False and var2 is not None:
            self.rename(var1.name+op_string[operation]+var2.name)
            self.user_name = False
            self.info['Formula'] = var1.info['Formula'] + \
                op_string[operation] + var2.info['Formula']
            self.info['Function']['variables'] += (var1, var2),
            self.info['Function']['operation'] += operation,
            ExperimentalValue.formula_register.update(
                {self.info["Formula"]: self.info["ID"]})
            self.info['Method'] += "Errors propagated by " +\
                self.error_method + ' method.\n'
            for root in var1.root:
                if root not in self.root:
                    self.root += var1.root
            for root in var2.root:
                if root not in self.root:
                    self.root += var2.root

            if var1.units == var2.units and op_string[operation] in ('+', '-'):
                self.units = var1.units

            elif op_string[operation] is '*':
                for key in var1.units:
                    self.units[key] = var1.units[key]
                for key in var2.units:
                    if key in var1.units:
                        self.units[key] = var1.units[key] + var2.units[key]
                        if self.units[key] == 0:
                            del self.units[key]
                    else:
                        self.units[key] = var2.units[key]

            elif op_string[operation] is '/':
                for key in var1.units:
                    self.units[key] = var1.units[key]
                for key in var2.units:
                    if key in var1.units:
                        self.units[key] = var1.units[key] - var2.units[key]
                        if self.units[key] == 0:
                            del self.units[key]
                    else:
                        self.units[key] = -var2.units[key]

        elif func_flag == True and var2 is None:
            self.rename(op_string[operation]+'('+var1.name+')')
            self.user_name = False
            self.info['Formula'] = op_string[operation] + '(' + \
                var1.info['Formula'] + ')'
            self.info['Function']['variables'] += (var1,),
            self.info['Function']['operation'] += operation,
            self.info['Method'] += "Errors propagated by " + \
                self.error_method + ' method.\n'
            ExperimentalValue.formula_register.update(
                {self.info["Formula"]: self.info["ID"]})
            for root in var1.root:
                if root not in self.root:
                    self.root += var1.root
            self.units = ''

        else:
            #TODO double check with Connor, but I think it was a bug above and we have to check == True
            # not is True, since 1 could also be True...
            print('Something went wrong in update_info')

###############################################################################
# Operations on measurement objects
###############################################################################

    ###########################################################################
    # ARITHMETIC OPERATIONS
    # Called whenever an operation (+, -, /, *, **) is encountered
    # Call operation_wrap() in error_operations.py
    ###########################################################################

    def __add__(self, other):
        '''Handles addition with Measurements.

        Ensures that the two objects can be added and then sends them to 
        error_operations.operation_wrap, which handles the addition and
        error propagation.
        '''
        #TODO: is this the correct implementation??? or should ARRAy create an ndarray???
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.add, self, other[i])
                #result.append(op.operation_wrap(op.add, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other+self
        else:
            return op.operation_wrap(op.add, self, other)

    def __radd__(self, other):
        '''Handles addition with Measurements.

        Ensures that the two objects can be added and then sends them to 
        error_operations.operation_wrap, which handles the addition and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.add, self, other[i])
                #result.append(op.operation_wrap(op.add, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other+self
        else:
            return op.operation_wrap(op.add, self, other)

    def __mul__(self, other):
        '''Handles multiplication with Measurements.

        Ensures that the two objects can be multiplied and then sends them to 
        error_operations.operation_wrap, which handles the multiplication and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.mul, self, other[i])
                #result.append(op.operation_wrap(op.mul, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other*self
        else:
            return op.operation_wrap(op.mul, self, other)

    def __rmul__(self, other):
        '''Handles multiplication with Measurements.

        Ensures that the two objects can be multiplied and then sends them to 
        error_operations.operation_wrap, which handles the multiplication and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.mul, self, other[i])
                #result.append(op.operation_wrap(op.mul, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other*self
        else:
            return op.operation_wrap(op.mul, self, other)

    def __sub__(self, other):
        '''Handles subtraction with Measurements.

        Ensures that the object can be subtracted and then sends them to 
        error_operations.operation_wrap, which handles the subtraction and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.sub, self, other[i])
                #result.append(op.operation_wrap(op.sub, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self-other
        else:
            return op.operation_wrap(op.sub, self, other)

    def __rsub__(self, other):
        '''Handles subtraction with Measurements.

        Ensures that the object can be subtracted and then sends them to 
        error_operations.operation_wrap, which handles the subtraction and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.sub, other[i], self)
                #result.append(op.operation_wrap(op.sub, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other-self
        else:
            return op.operation_wrap(op.sub, other, self)

    def __truediv__(self, other):
        '''Handles division with Measurements.

        Ensures that the object can be divided by and then sends them to 
        error_operations.operation_wrap, which handles the division and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.div, self, other[i])
                #result.append(op.operation_wrap(op.div, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self/other
        else:
            return op.operation_wrap(op.div, self, other)

    def __rtruediv__(self, other):
        '''Handles division with Measurements.

        Ensures that the object can be divided by and then sends them to 
        error_operations.operation_wrap, which handles the division and
        error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.div, self, other[i])
                #result.append(op.operation_wrap(op.div, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other/self
        else:
            return op.operation_wrap(op.div, other, self)

    def __pow__(self, other):
        '''Handles exponentiation with Measurements.

        Ensures that the object can be raised to the power of the other
        and then sends them to error_operations.operation_wrap, which
        handles the exponentiation and error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.power, self, other[i])
                #result.append(op.operation_wrap(op.power, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self**other
        else:
            return op.operation_wrap(op.power, self, other)

    def __rpow__(self, other):
        '''Handles exponentiation with Measurements.

        Ensures that the object can be raised to the power of the other
        and then sends them to error_operations.operation_wrap, which
        handles the exponentiation and error propagation.
        '''
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.power, self, other[i])
                #result.append(op.operation_wrap(op.power, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other**self
        else:
            return op.operation_wrap(op.power, other, self)

    # Calls neg() in error_operations, which returns the negative of the value
    def __neg__(self):
        '''Returns the negative of a Measurement object.
        '''
        import qexpy.error_operations as op
        return op.neg(self)

    # Returns the length of the ExperimentalValue
    def __len__(self):
        '''Returns the length of a the array used to create the Measurement object.'''
        return self.info['Data'].size

    ###########################################################################
    # COMPARISON OPERATIONS
    # Called whenever a comparison (>, <, >=, ==, ...) is made
    # Makes the relevant comparison and return a boolean
    ###########################################################################
    def __eq__(self, other):
        '''Checks if two Measurements are the same.

        Returns True if the means of the two Measurements are the same.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean == other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean == other.mean

    def __req__(self, other):
        '''Checks if two Measurements are the same.

        Returns True if the means of the two Measurements are the same.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean == other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean == other.mean
            
    def __gt__(self, other):
        '''Checks if a Measurement is greater than another Measurement.

        Returns True if the mean of the Measurement is greater than the mean
        of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean > other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean > other.mean
            
    def __rgt__(self, other):
        '''Checks if a Measurement is less than another Measurement.

        Returns True if the mean of the Measurement is less than the mean
        of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean < other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean < other.mean 
            
    def __ge__(self, other):
        '''Checks if a Measurement is greater than or equal to another Measurement.

        Returns True if the mean of the Measurement is greater than or equal to
        the mean of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean >=other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean >= other.mean
            
    def __rge__(self, other):
        '''Checks if a Measurement is less than or equal to another Measurement.

        Returns True if the mean of the Measurement is less than or equal to
        the mean of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean <= other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean <= other.mean
            
    def __lt__(self, other):
        '''Checks if a Measurement is less than another Measurement.

        Returns True if the mean of the Measurement is less than the mean
        of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean < other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean < other.mean
            
    def __rlt__(self, other):
        '''Checks if a Measurement is greater than another Measurement.

        Returns True if the mean of the Measurement is greater than the mean
        of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean > other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean > other.mean
            
    def __le__(self, other):
        '''Checks if a Measurement is less than or equal to another Measurement.

        Returns True if the mean of the Measurement is less than or equal to
        the mean of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean <= other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean <= other.mean
            
    def __rle__(self, other):
        '''Checks if a Measurement is greater than or equal to another Measurement.

        Returns True if the mean of the Measurement is greater than or equal to
        the mean of the other Measurement.
        '''
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean >= other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean >= other.mean      

###############################################################################
# Miscellaneous Methods
###############################################################################

    def _check_der(self, b):
        '''
        Checks for a derivative with respect to b, else zero is defined as
        the derivative.

        Checks the existance of the derivative of an object in the
        dictionary of said object with respect to another variable, given
        the variable itself, then checking for the ID of said variable
        in the .derivative dictionary. If non exists, the deriviative is
        assumed to be zero.
        '''
        for key in b.derivative:
            if key in self.derivative:
                pass
            else:
                self.derivative[key] = 0


###############################################################################
# Measurement Sub-Classes
###############################################################################


class Measurement(ExperimentalValue):
    '''
    Subclass of ExperimentalValue, specified by the user and treated as variables
    or arguments of functions when propagating error.
    '''
    id_number = 0

    def __init__(self, *args, name=None, units=None):
        super().__init__(*args, name=name)

        if name is not None:
            self.name = name
        else:
            self.name = 'unnamed_var%d' % (Measurement.id_number)

        if units is not None:
            if type(units) is str:
                self.units[units] = 1
            else:
                for i in range(len(units)//2):
                    self.units[units[2*i]] = units[2*i+1]

        self.type = "ExperimentalValue"
        self.info['ID'] = 'var%d' % (Measurement.id_number)
        self.info['Formula'] = 'var%d' % (Measurement.id_number)
        Measurement.id_number += 1
        self.derivative = {self.info['ID']: 1}
        self.covariance = {self.info['ID']: self.std**2}
        self.correlation = {self.info['ID']: 1}
        ExperimentalValue.register.update({self.info["ID"]: self})
        self.root = (self.info["ID"],)
        self.der = [self.mean, self.std]
        self.MC = [self.mean, self.std]
        self.MinMax = [self.mean, self.std]


class Function(ExperimentalValue):
    '''
    Subclass of ExperimentalValue, which are measurements created by operations or
    functions of other measurement type objects.
    '''
    id_number = 0

    def __init__(self, *args, name=None):
        super().__init__(*args, name=name)

        self.name = 'obj%d' % (Function.id_number)
        self.info['ID'] = 'obj%d' % (Function.id_number)
        self.type = "Function"
        Function.id_number += 1
        self.derivative = {self.info['ID']: 1}
        ExperimentalValue.register.update({self.info["ID"]: self})
        self.covariance = {self.info['ID']: self.std**2}
        self.correlation = {self.info['ID']: 1}
        self.root = ()
        self.der = None
        #These are set by super()
        #self.MC = None
        #self.MinMax = None
        self.error_flag = False


class Constant(ExperimentalValue):
    '''
    Subclass of ExperimentalValue, not neccesarily specified by the user,
    called when a consant (int, float, etc.) is used in operation with a
    measurement. This class is called before calculating operations to
    ensure objects can be combined. The mean of a constant is the specified
    value, the standard deviation is zero, and the derivarive with respect
    to anything is zero.
    '''
    def __init__(self, arg, name=None, units=None):
        super().__init__(arg, 0)

        if name is not None:
            self.name = name
        else:
            self.name = '%d' % (arg)

        if units is not None:
            if type(units) is str:
                self.units[units] = 1
            else:
                for i in range(len(units)//2):
                    self.units[units[2*i]] = units[2*i+1]

        self.info['ID'] = 'Constant'
        self.info["Formula"] = '%f' % arg
        self.derivative = {}
        self.info["Data"] = np.array([arg])
        self.type = "Constant"
        self.covariance = {self.name: 0}
        self.root = ()


class Measurement_Array(np.ndarray):
    ''' A numpy-based array of Measurement objects'''
    id_number = 0
    _error_method = "Derivative"
    def __new__(subtype, shape, dtype=Measurement, buffer=None, offset=0,
          strides=None, order=None, name = None, units = None, error_method='Derivative'):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        if name is not None:
            obj.name = name
        else:
            obj.name = 'unnamed_arr%d' % (Measurement_Array.id_number)

        obj.error_method = error_method

        obj.units = {}
        if units is not None:
            if type(units) is str:
                obj.units[units] = 1
            else:
                for i in range(len(units)//2):
                    obj.units[units[2*i]] = units[2*i+1]

        Measurement_Array.id_number += 1
        
        return obj

    def __array_finalize__(self, obj):
        '''Sets the name and units of the MeasurementArray during creation.
        '''
        if obj is None: return
        self.units = getattr(obj, 'units', None)
        self.name = getattr(obj, 'name', None)
        
    def __array_wrap__(self, out_arr, context=None):
        '''Used to make sure that numpy functions work on MeasurementArrays
        and they return MeasurementArrays.
        '''
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def error_method(self):
        '''Returns the method (Monte Carlo, derivative or min max) 
        used to calculate error of a MeasurementArray object.
        '''
        return self._error_method

    @error_method.setter
    def error_method(self, method):
        '''Sets the method (Monte Carlo, derivative or min max) 
        used to calculate error of a MeasurementArray object.
        '''
        mc_list = ('MC', 'mc', 'montecarlo', 'Monte Carlo', 'MonteCarlo',
                   'monte carlo',)
        min_max_list = ('Min Max', 'MinMax', 'minmax', 'min max',)
        derr_list = ('Derivative', 'derivative', 'diff', 'der', 'Default',
                     'default',)

        if method in mc_list:
            self._error_method = "Monte Carlo"
        elif method in min_max_list:
            self._error_method = "Min Max"
        elif method in derr_list:
            self._error_method = "Derivative"
        else:
            print("Method not recognized, using derivative method.")
            self._error_method = "Derivative"

    @property
    def means(self):
        '''Returns a numpy array with the means of each value in the MeasurementArray,
        as calculated by the method (der, MC, MinMax).
        '''
        if self.size == 0:
            return np.ndarray(0)

        means = np.ndarray(shape=self.shape)

        for index, item in np.ndenumerate(self):
            if item is not None:
                if self.error_method == "MinMax":
                    means[index] = item.MinMax[0]
                elif self.error_method == "MC":
                    means[index] = item.MC[0]
                else:
                    means[index] = item.mean
            else:
                means[index] = 0
        return means

    @property
    def stds(self):
        '''Returns a numpy array with the standard deviations of each value
        in the MeasurementArray, as calculated by the method (der, MC, MinMax).
        '''
        if self.size == 0:
            return np.ndarray(0)
        
        stds = np.ndarray(shape=self.shape)
        
        for index, item in np.ndenumerate(self):
            if item is not None:              
                if self.error_method == "MinMax":
                    stds[index] = 9
                elif self.error_method == "MC":
                    stds[index] = item.MC[1]
                else:
                    stds[index] = item.std
            else:
                stds[index] = 0
        return stds

    @stds.setter
    def stds(self, error):
        '''Sets the standard deviations of each value in the MeasurementArray,
        either to the same value for all Measurements or to a different value
        for each Measurement.
        '''
        n = self.size

        if isinstance(error, qu.number_types):#MA([,,,,], error = x)
            for i in range(n):
                self[i].std=error

        elif isinstance(error, qu.array_types):#MA([,,,,], error = [])
            if len(error)==n:#MA([,,,,], error = [,,,,])
                for i in range(n):
                    self[i].std=error[i]

            elif len(error)==1: #MA([,,,,], error = [x])
                for i in range(n):
                    self[i].std=error[0]
            else:
                print("Error list must be the same length as the original list")

    @property
    def mean(self):
        '''Returns the mean of the means of the Measurements in the MeasurementArray.
        '''
        nparr = self.means
        self._mean = nparr.mean()
        return self._mean

    @property
    def error_weighted_mean(self):
        '''Returns the error weighted mean and error of a MeasurementArray.

        The error weighted mean gives more weight to more precise Measurements.
        '''
        means = self.means
        stds = self.stds
        stds2 = stds**2
        sumw2=0
        mean=0

        #Needs to be a loop to catch the case of std == 0
        for i in range(means.size):
            if stds[i] == 0.0:
                continue
            w2 = 1./stds2[i]
            mean += w2*means[i]
            sumw2 += w2

        self._error_weighted_mean =  (0. if sumw2==0 else mean/sumw2)
        self._error_weighted_std =  (0. if sumw2==0 else np.sqrt(1./sumw2))

        return Measurement(self._error_weighted_mean, self._error_weighted_std)
    
    def std(self, ddof=1, method="der"):
        '''Return standard deviation of the means of each measurement'''
        nparr = self.means
        return nparr.std(ddof=ddof)
    
    def get_units_str(self):
        '''Returns a string representation of the units.
        '''
        unit_string = ''
        if self.units != {}:
            for key in self.units:
                if self.units[key] == 1 and len(self.units.keys()) is 1:
                    unit_string = key + unit_string
                else:
                    unit_string += key+'^%d' % (self.units[key])
                    unit_string += ' '

                if unit_string == '':
                    unit_string = 'unitless'
        return unit_string

    def set_units(self, units=None):
        '''Sets the units of a MeasurementArray.
        '''
        if units is not None:
            for mes in self:
                if type(units) is str:
                    mes.units[units] = 1
                else:
                    for i in range(len(units)//2):
                        mes.units[units[2*i]] = units[2*i+1]
             
            if type(units) is str:
                self.units[units] = 1
            else:
                for i in range(len(units)//2):
                    self.units[units[2*i]] = units[2*i+1]
    
    def __str__(self):
        '''Returns a string representation of the MeasurementArray.
        '''
        theString=''
        for i in range(self.size):
            theString += self[i].__str__()
            if i != self.size-1:
                theString += ',\n'
        return theString
            
def MeasurementArray(data, error=None, name=None, units=None, error_method='Derivative'):
    '''Function to easily construct a Measurement_Array object.
    '''

    array = Measurement_Array(0, name=name, units=units, error_method='Derivative')
    user_name= True if name != None else False
        
    if error is None: #MA(data)
        if isinstance(data, Measurement_Array):# MA(MA)
            #TODO should this be a copy?? This will just 
            #make array be a reference to data...
            array = data
            array.set_units(units)
            #allow the name to be updated:
            if name is not None:
                array.name = name  
                
        elif isinstance(data, qu.array_types): #MA([...])
            n = len(data)       
            array.resize(n)
            if isinstance(data[0], qu.array_types) and len (data[0]) == 2: #MA([ (,), (,), (,)])
                for i in range(n):
                    data_name = "{}_{}".format(array.name,i)
                    array[i]=Measurement(data[i][0],data[i][1], units=units, name=data_name)
                    array[i].user_name=user_name
                    
            elif isinstance(data[0], qu.number_types): #MA([,,,])
                for i in range(n):
                    data_name = "{}_{}".format(array.name,i)
                    array[i]=Measurement(float(data[i]),0., units=units, name=data_name)
                    array[i].user_name=user_name
            else:
                print("unsupported type for data")
             
        elif isinstance(data, qu.int_types): #MA(n)
            array.resize(data)
            for i in range(data):
                data_name = "{}_{}".format(array.name,i)
                array[i]=Measurement(0.,0., units=units, name=data_name)
                array[i].user_name=user_name
        else:
            print("unsupported type for data")
            
    else: #error is not None
        if isinstance(data, Measurement_Array):
            array = data
            array.set_errors(error)
            array.set_units(units)
            #allow the name to be updated:
            if name is not None:
                array.name = name

            
        elif isinstance(data, qu.array_types): #MA([], error = ...)
            n = len(data)       
            array.resize(n)
            
            if isinstance(data[0], qu.number_types):#MA([,,,,], error = ...)
                
                if isinstance(error, qu.number_types):#MA([,,,,], error = x)
                    for i in range(n):
                        data_name = "{}_{}".format(array.name,i)
                        array[i]=Measurement(float(data[i]),error, units=units, name=data_name)
                        array[i].user_name=user_name
                elif isinstance(error, qu.array_types):#MA([,,,,], error = [])
                    if len(error)==len(data):#MA([,,,,], error = [,,,,])
                        for i in range(n):
                            data_name = "{}_{}".format(array.name,i)
                            array[i]=Measurement(float(data[i]),error[i], units=units, name=data_name)    
                            array[i].user_name=user_name
                    elif len(error)==1: #MA([,,,,], error = [x])
                        for i in range(n):
                            data_name = "{}_{}".format(array.name,i)
                            array[i]=Measurement(float(data[i]),error[0], units=units, name=data_name)
                            array[i].user_name=user_name
                    else:
                        print("error array must be same length as data")
                else:
                    print("unsupported type for error")
                    
            else: # data[0] must be a float
                print("unsupported type for data:", type(data[0]))
                
        elif isinstance(data, qu.number_types): #MA(x,error=...)
            array.resize(1)
            if isinstance(error, qu.number_types):#MA(x, error = y)
                data_name = "{}_{}".format(array.name,0)
                array[0]=Measurement(float(data),error, units=units, name=data_name)
                array[0].user_name=user_name
            elif isinstance(error, qu.array_types) and len(error)==1:#MA(x, error = [u])
                data_name = "{}_{}".format(array.name,0)
                array[0]=Measurement(float(data),error[0], units=units, name=data_name)
                array[0].user_name=user_name
            else:
                print("unsupported type for error")
        else:
            print("unsupported type of data")
        
    return array        
        
        

###############################################################################
# Mathematical Functions
# These are called for functions in the form: error.func(ExperimentalValue)
# They call operation_wrap() in the error_operations.py file
###############################################################################

ExperimentalValue.ARRAY = ExperimentalValue.ARRAY +(Measurement_Array,)

def sqrt(x):
    import qexpy.error_operations as op      
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sqrt, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sqrt, x[index], func_flag=True)  
        return result
    else:
        return op.operation_wrap(op.sqrt, x, func_flag=True)

def sin(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return np.ndarray(0, dtype=float)
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sin, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sin, x[index], func_flag=True)    
        #result = []      
        #for value in x:
            #result.append(op.operation_wrap(op.sin, value, func_flag=True))
        
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.sin(x)
    else:
        return op.operation_wrap(op.sin, x, func_flag=True)


def cos(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
          
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.cos, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.cos, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.cos, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.cos(x)
    else:
        return op.operation_wrap(op.cos, x, func_flag=True)


def tan(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.tan, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.tan, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.tan, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.tan(x)
    else:
        return op.operation_wrap(op.tan, x, func_flag=True)


def sec(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sec, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.sec, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.sec, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.cos(x)
    else:
        return op.operation_wrap(op.sec, x, func_flag=True)


def csc(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.csc, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.csc, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.csc, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.sin(x)
    else:
        return op.operation_wrap(op.csc, x, func_flag=True)


def cot(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.cot, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.cot, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.cot, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.tan(x)
    else:
        return op.operation_wrap(op.cot, x, func_flag=True)


def log(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.log, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.log, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.log, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.log(x)
    else:
        return op.operation_wrap(op.log, x, func_flag=True)


def exp(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.exp, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.exp, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.exp, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.exp(x)
    else:
        return op.operation_wrap(op.exp, x, func_flag=True)


def e(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.exp, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.exp, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.exp, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.exp(x)
    else:
        return op.operation_wrap(op.exp, x, func_flag=True)


def asin(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.asin, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.asin, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.asin, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.asin(x)
    else:
        return op.operation_wrap(op.asin, x, func_flag=True)


def acos(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.acos, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.acos, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.acos, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.acos(x)
    else:
        return op.operation_wrap(op.acos, x, func_flag=True)


def atan(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        if len(x) <1:
            return []
        if isinstance(x[0],ExperimentalValue):
            result = Measurement_Array(len(x))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.atan, x[index], func_flag=True)
        else:
            result = np.ndarray(len(x), dtype=type(x[0]))
            for index in range(len(x)):
                result[index]=op.operation_wrap(op.atan, x[index], func_flag=True)
        #result = []
        #for value in x:
        #    result.append(op.operation_wrap(op.atan, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.atan(x)
    else:
        return op.operation_wrap(op.atan, x, func_flag=True)


###############################################################################
# Printing Methods
###############################################################################


def set_print_style(style=None, sigfigs=None):
    '''Change style ("default","latex","scientific") of printout for
    Measurement objects.

    The default style prints as the user might write a value, that is
    'x = 10 +/- 1'.

    Latex style prints in the form of 'x = (10\pm 1)\e0' which is ideal for
    pasting values into a Latex document as will be the case for lab reports.

    The scientific style prints the value in reduced scientific notation
    such that the error is a single digit, 'x = (10 +/- 1)*10**0'.
    '''
    latex = ("Latex", "latex", 'Tex', 'tex',)
    Sci = ("Scientific", "Sci", 'scientific', 'sci', 'sigfigs',)
    Default = ('default', 'Default',)

    if sigfigs is not None:
        set_sigfigs(sigfigs)
        ExperimentalValue.figs_on_uncertainty = False

    if style in latex:
        ExperimentalValue.print_style = "Latex"
    elif style in Sci:
        ExperimentalValue.print_style = "Scientific"
    elif style in Default:
        ExperimentalValue.print_style = "Default"
    else:
        print('''A style must be a string of either: Scientific notation,
        Latex, or the default style. Using default.''')
        ExperimentalValue.print_style = "Default"


def set_error_method(chosen_method):
    '''Choose the method of error propagation to be used. Enter a string.

    Function to change default error propogation method used in
    measurement functions.
    '''
    mc_list = (
        'MC', 'mc', 'montecarlo', 'Monte Carlo', 'MonteCarlo',
        'monte carlo',)
    min_max_list = ('Min Max', 'MinMax', 'minmax', 'min max',)
    derr_list = ('Derivative', 'derivative', 'diff', 'der', 'Default',
                 'default',)

    if chosen_method in mc_list:
        ExperimentalValue._error_method = "Monte Carlo"
    elif chosen_method in min_max_list:
        ExperimentalValue._error_method = "Min Max"
    elif chosen_method in derr_list:
        ExperimentalValue._error_method = "Derivative"
    else:
        print("Method not recognized, using derivative method.")
        ExperimentalValue._error_method = "Derivative"


def set_sigfigs_error(sigfigs=3):
    '''Change the number of significant figures shown in print()
    based on the number of sig figs in the error
    '''
    if type(sigfigs) is None:
        ExperimentalValue.figs = None
    elif type(sigfigs) is int and sigfigs > 0:
        ExperimentalValue.figs = sigfigs
        ExperimentalValue.figs_on_uncertainty = True
    else:
        raise TypeError('''Specified number of significant figures must be
                        and interger greater than zero.''')


def set_sigfigs_centralvalue(sigfigs=3):
    '''Change the number of significant figures shown in print()
    based on the number of sig figs in the central value
    '''
    if type(sigfigs) is None:
        ExperimentalValue.figs = None
    elif sigfigs > 0 and type(sigfigs) is int:
        ExperimentalValue.figs = sigfigs
        ExperimentalValue.figs_on_uncertainty = False
    else:
        raise TypeError('''Specified number of significant figures must be
                        and interger greater than zero.''')


def set_sigfigs(sigfigs=3):
    '''Change the number of significant figures shown in print()
    based on the number of sig figs in the error
    '''
    set_sigfigs_error(sigfigs)


def _return_exponent(value):
    '''Returns the exponent of the argument in reduced scientific notation.
    '''
    value = abs(value)
    flag = True
    i = 0

    while(flag):
        if value == 0:
            flag = False
        elif value == float('inf'):
            return float("inf")
        elif value < 1:
            value *= 10
            i -= 1
        elif value >= 10:
            value /= 10
            i += 1
        elif value >= 1 and value < 10:
            flag = False
    return i


def _return_print_values(variable, method):
    '''Function for returning the correct mean and std for the method
    selected.
    '''
    if isinstance(variable, Constant):
        return (variable.mean, 0,)
    if ExperimentalValue._error_method == 'Derivative':
        [mean, std] = variable.der
    elif ExperimentalValue._error_method == 'Monte Carlo':
        [mean, std] = variable.MC
    elif ExperimentalValue._error_method == 'Min Max':
        [mean, std] = variable.MinMax

    if method is not None:
        if method is 'Derivative':
            [mean, std] = variable.der
        elif method is 'Monte Carlo':
            [mean, std] = variable.MC
        elif method is 'Min Max':
            [mean, std] = variable.MinMax
    return (mean, std,)


def _tex_print(self, method=None):
    '''Creates string used by __str__ in a style useful for printing in Latex,
    as a value with error, in brackets multiplied by a power of ten. (ie.
    15+/-0.3 is (150 \pm 3)\e-1. Where Latex parses \pm as +\- and \e as
    *10**-1)
    '''
    mean, std = _return_print_values(self, method)

    if ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == False:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(mean)
        mean = int(round(mean*10**(figs - i), 0))
        std = int(round(std*10**(figs - i), 0))

        if i - figs != 0:
            return "(%d \pm %d)*10^{%d}" % (mean, std, i - figs)
        else:
            return "(%d \pm %d)" % (mean, std)

    elif ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == True:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(std)
        mean = int(round(mean*10**(figs - i), 0))
        std = int(round(std*10**(figs - i), 0))

        if i - figs != 0:
            return "(%d \pm %d)*10^{%d}" % (mean, std, i - figs)
        else:
            return "(%d \pm %d)" % (mean, std)

    else:
        i = _return_exponent(std)
        mean = int(round(mean*10**-i, 0))
        std = int(round(std*10**-i, 0))

        if i != 0:
            return "(%d \pm %d)*10^{%d}" % (mean, std, i)
        else:
            return "(%d \pm %d)" % (mean, std)


def _def_print(self, method=None):
    '''Returns string used by __str__ as two numbers representing mean and error
    to either the first non-zero digit of error or to a specified number of
    significant figures.
    '''
    method = self.error_method if not method else method
    mean, std = _return_print_values(self, method)

    if ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == False:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(mean)

        decimal_places = figs - i
        if decimal_places > 0:
            n = '%d' % (decimal_places)
            n = "%."+n+"f"
        else:
            n = '%.0f'
        std = float(round(std, decimal_places))
        mean = float(round(mean, decimal_places))
        return n % (mean)+" +/- "+n % (std)

    elif ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == True:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(std)

        decimal_places = figs - i
        if decimal_places > 0:
            n = '%d' % (decimal_places)
            n = "%."+n+"f"
        else:
            n = '%.0f'
        std = float(round(std, decimal_places))
        mean = float(round(mean, decimal_places))
        return n % (mean)+" +/- "+n % (std)

    else:
        if mean == float('inf') and std == float('inf'):
            return "inf +/- inf"
        
        if mean == float('inf'):
            return "inf"
                        
        i = _return_exponent(std)

        if i < 0:
            n = '%d' % (-i)
            n = "%."+n+"f"
        else:
            n = '%.0f'
            
        mean = float(round(mean, -i))
        if std == float('inf'):
            return  n % (mean)+" +/- inf"
        
        std = float(round(std, -i))
        
        return n % (mean)+" +/- "+n % (std)


def _sci_print(self, method=None):
    '''Returns string used by __str__ as two numbers representing mean and
    error, each in scientific notation to a specified numebr of significant
    figures, or 3 if none is given.
    '''
    mean, std = _return_print_values(self, method)

    if ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == False:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(mean)
        mean = int(round(mean*10**(figs - i), 0))
        std = int(round(std*10**(figs - i), 0))

        if i - figs != 0:
            return "(%d +/- %d)*10^(%d)" % (mean, std, i - figs)
        else:
            return "(%d +/- %d)" % (mean, std)

    elif ExperimentalValue.figs is not None and\
            ExperimentalValue.figs_on_uncertainty == True:

        if mean == float('inf'):
            return "inf"

        figs = ExperimentalValue.figs - 1
        i = _return_exponent(std)
        mean = int(round(mean*10**(figs - i), 0))
        std = int(round(std*10**(figs - i), 0))

        if i - figs != 0:
            return "(%d +/- %d)*10^(%d)" % (mean, std, i - figs)
        else:
            return "(%d +/- %d)" % (mean, std)

    else:
        i = _return_exponent(std)
        mean = int(round(mean*10**-i, 0))
        std = int(round(std*10**-i, 0))

        if i != 0:
            return "(%d +/- %d)*10^(%d)" % (mean, std, i)
        else:
            return "(%d +/- %d)" % (mean, std)


###############################################################################
# Random Methods
###############################################################################


def show_histogram(data, title=None, output='inline'):
    '''Creates a histogram of the inputted data using Bokeh or mpl.
    '''
    if type(data) not in ARRAY:
        print('''Input histogram data must be an array''')
        return

    if type(title) is str:
        hist_title = title
    elif title is None:
        hist_title = 'Histogram'
    else:
        print('Histogram title must be a string.')
        hist_title = 'Histogram'

    mean, std = _variance(data)

    xy_data = q.XYDataSet(xdata = data, is_histogram = True, data_name=hist_title)
    fig = q.MakePlot()
    fig.add_dataset(xy_data, color = color)
    fig.x_range = [min(data)*.95,max(data)*1.05]
    fig.y_range = [0,max(xy_data.ydata)*1.2]

    # Draws lines at the mean and location of the mean +/- standard deviation.
    mean = self.mean
    std = self.std
    fig.add_line(x=mean, dashed=False, color='red')
    fig.add_line(x=mean+std, dashed=True, color='red')
    fig.add_line(x=mean-std, dashed=True, color='red')

    fig.show()
    return fig


def numerical_partial_derivative(func, var, *args):
    '''Returns the parital derivative of a dunction with respect to var.

    This function wraps the inputted function to become a function
    of only one variable, the derivative is taken with respect to said
    variable.
    '''
    def restrict_dimension(x):
        partial_args = list(args)
        partial_args[var] = x
        return func(*partial_args)
    return numerical_derivative(restrict_dimension, args[var])


def numerical_derivative(function, point, dx=1e-10):
    '''Returns the first order derivative of a function.
    '''
    return (function(point+dx)-function(point))/dx


def _variance(*args, ddof=1):
    '''Returns a tuple of the mean and standard deviation of a data array.

    Uses a more sophisticated variance calculation to speed up calculation of
    mean and standard deviation.
    '''
    args = args[0]
    Sum = 0
    SumSq = 0
    N = len(args)
    for i in range(N):
        Sum += args[i]
        SumSq += args[i]*args[i]

    std = ((SumSq-Sum**2/N)/(N-1))**(1/2)
    mean = Sum/N

    return (mean, std, )

def _weighted_variance(mean, std, ddof=1):
    '''Calculates the variance weighted mean and standard deviation.
    '''
    from math import sqrt

    w = np.power(std, -2)
    w_mean = sum(np.multiply(w, mean))/sum(w)
    w_std = 1/sqrt(sum(w))
    return (w_mean, w_std)


def reset_variables():
    '''Resets the ID number, directories and methods to their original values.
    Useful in Jupyter Notebooks if variables were unintentionally repeated.
    '''
    Measurement.id_number = 0
    Function.id_number = 0
    ExperimentalValue.register = {}
    ExperimentalValue.formula_register = {}
    ExperimentalValue.error_method = "Derivative"
    ExperimentalValue.mc_trial_number = 10000
    ExperimentalValue.print_style = "Default"
    ExperimentalValue.figs = 3
