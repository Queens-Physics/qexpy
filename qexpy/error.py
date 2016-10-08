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
    '''
    Root class of objects which containt a mean and standard deviation.
    From this class, objects with properties pertaining to their use or
    formulation can be instanced. (ie. the result of an operation of
    measured values, called Funciton and Measured respectivly)
    '''
    error_method = "Derivative"  # Default error propogation method
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
        '''
        Creates a variable that contains a mean, standard deviation,
        and name for inputted data.
        '''
        if len(args) ==1:
            if isinstance(args[0], qu.array_types):
                data = np.ndarray(len(args[0]))
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
                self.std = float(args[1])
                data = np.ndarray(1)
                error_data = np.ndarray(1)
                data[0] = self.mean
            else:
                raise TypeError('''Input must be either a single array of values,
                      or the central value and uncertainty in one measurement''')
        else:
            raise TypeError('''Input must be either a single array of values,
                  or the central value and uncertainty in one measurement''')
       
        self.der = [self.mean, self.std]
        self.MinMax = [self.mean, self.std]
        self.MC = [self.mean, self.std]
        
        self.info = {
                'ID': '', 'Formula': '', 'Method': '', 'Data': data,\
                'Function': {
                        'operation': (), 'variables': ()}, }

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
        '''
        Method called when printing measurement objects.
        '''
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
            string = _tex_print(self, method=self.MC)
        elif self.print_style == "Default":
            string = _def_print(self, method=self.MC)
        elif self.print_style == "Scientific":
            string = _sci_print(self, method=self.MC)
        print(string)

    def print_min_max_error(self):
        '''Prints the result of a Min-Max method error propagation.

        The purpose of this method is to easily compare the results of a
        Min-Max propagation with whatever method is chosen to confirm that
        the Min-Max is the upper bound of the error.
        '''
        if self.print_style == "Latex":
            string = _tex_print(self, method=self.MinMax)
        elif self.print_style == "Default":
            string = _def_print(self, method=self.MinMax)
        elif self.print_style == "Scientific":
            string = _sci_print(self, method=self.MinMax)
        print(string)

    def print_deriv_error(self):
        '''Prints the result of a Min-Max method error propagation.

        The purpose of this method is to easily compare the results of a
        Min-Max propagation with whatever method is chosen to confirm that
        the Min-Max is the upper bound of the error.
        '''
        if self.print_style == "Latex":
            string = _tex_print(self, method=self.Derivative)
        elif self.print_style == "Default":
            string = _def_print(self, method=self.Derivative)
        elif self.print_style == "Scientific":
            string = _sci_print(self, method=self.Derivative)
        print(string)

    def get_derivative(self, variable=None):
        '''
        Returns the numerical value of the derivative with respect to the
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
                  other than a Measuremnt is zero.''')
            return 0

        elif variable is None:
            raise TypeError('''The object must be differentiated with respect to another
            Measurement.''')

        if variable.info['ID'] not in self.derivative:
            self.derivative[variable.info['ID']] = 0

        derivative = self.derivative[variable.info["ID"]]
        return derivative

    def get_error(self):
        '''Returns the error associated the Measurement for whatever error
        propagation method is selected.
        '''
        return self.std
    
    def get_uncertainty(self):
        return self.get_error()
    
    def get_relative_error(self):
        return self.std/self.mean if self.mean !=0 else 0.
    
    def get_relative_uncertainty(self):
        return self.get_relative_error()
    
    def get_error_on_mean(self):
        '''Returns the error on the mean if the Measurement is from a set 
        of data'''
        if self.error_on_mean:
            return self.error_on_mean
        else:
            print("Error: error on mean not calculated")
            return 0
    
    def get_mean(self):
        ''' Returns the central value associated with the Measurement object
        using whatever error propagation method is selected.
        '''
        return self.mean

    def get_name(self):
        '''Returns the name of the associated object, whether user-specified
        or auto-generated.
        '''
        return self.name

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

    def get_data_array(self):
        '''Returns the array of data used to create this Measurement.
        '''
        if self.info['Data'] is None:
            print('No data array exists.')
            return None
        return self.info['Data']
    
    def show_histogram(self, bins=50, title=None, output='inline'):
        '''Creates a histogram of the inputted data using Bokeh.
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
            
        hist, edges = np.histogram(self.info['Data'], bins=bins)
        
        if q.plot_engine in q.plot_engine_synonyms["mpl"]:              
            p = q.MakePlot(q.XYDataSet(self.info['Data'],
                                     data_name=hist_title,
                                     is_histogram=True, bins=bins))
            p.datasets_colors[-1]='blue'
            
            p.x_range_margin=edges[1]-edges[0]
            p.y_range_margin=hist.max()*0.2
            p.y_range[0]=0.
            
            p.mpl_plot([self.mean]*2, [0, hist.max()*1.1],color='red',lw=2)
            p.mpl_plot([self.mean-self.std]*2, [0, hist.max()], color='red',
                ls='--', lw=2)
            p.mpl_plot([self.mean+self.std]*2, [0, hist.max()], color='red',
                ls='--', lw=2)
            
            p.show(refresh=False)
            return p
        else:
        
            p1 = bp.figure(title=hist_title, tools='save, pan, box_zoom, wheel_zoom, reset',
                    background_fill_color="#FFFFFF")

            p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#036564", line_color="#033649")

            p1.line([self.mean]*2, [0, hist.max()*1.1], line_color='red')
            p1.line([self.mean-self.std]*2, [0, hist.max()], line_color='red',
                line_dash='dashed')
            p1.line([self.mean+self.std]*2, [0, hist.max()], line_color='red',
                line_dash='dashed')
        
            if output =='file' or not qu.in_notebook():
                bi.output_file(self.name+' histogram.html', title=hist_title)
            elif not qu.bokeh_ouput_notebook_called:
                bi.output_notebook()
                #This must be the first time calling output_notebook,
                #keep track that it's been called:
                qu.bokeh_ouput_notebook_called = True
            
            bp.show(p1)
            return p1
    
    def show_MC_histogram(self, bins=50, title=None, output='inline'):
        '''Creates and shows a Bokeh plot of a histogram of the values
        calculated by a Monte Carlo error propagation.
        '''
        if self.MC_list is None:
            print("no MC data to histogram")
            return None

        if type(title) is str:
            hist_title = title
        elif title is None:
            hist_title = self.name+' Histogram'
        else:
            print('Histogram title must be a string.')
            hist_title = self.name+' Histogram'

        p1 = bp.figure(title=hist_title, tools='''save, pan, box_zoom,
                       wheel_zoom, reset''',
                       background_fill_color="#FFFFFF")

        hist, edges = np.histogram(self.MC_list, bins=bins)
        
        if q.plot_engine in q.plot_engine_synonyms["mpl"]:              
            p = q.MakePlot(q.XYDataSet(self.MC_list,
                                     data_name=hist_title,
                                     is_histogram=True, bins=bins))
            p.datasets_colors[-1]='blue'
            
            p.x_range_margin=edges[1]-edges[0]
            p.y_range_margin=hist.max()*0.2
            p.y_range[0]=0.
            
            p.mpl_plot([self.mean]*2, [0, hist.max()*1.1],color='red',lw=2)
            p.mpl_plot([self.mean-self.std]*2, [0, hist.max()], color='red',
                ls='--', lw=2)
            p.mpl_plot([self.mean+self.std]*2, [0, hist.max()], color='red',
                ls='--', lw=2)
            
            p.show(refresh=False)
            return p
        else:
            p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                    fill_color="#036564", line_color="#033649")

            p1.line([self.mean]*2, [0, hist.max()*1.1], line_color='red')
            p1.line([self.mean-self.std]*2, [0, hist.max()], line_color='red',
                    line_dash='dashed')
            p1.line([self.mean+self.std]*2, [0, hist.max()], line_color='red',
                line_dash='dashed')
    
            if output == 'file' or not qu.in_notebook():
                bi.output_file(self.name+' histogram.html', title=hist_title)
            elif not qu.bokeh_ouput_notebook_called:
                bi.output_notebook()
                # This must be the first time calling output_notebook,
                # keep track that it's been called:
                qu.bokeh_ouput_notebook_called = True

            bp.show(p1)
        return p1

###############################################################################
# Methods for Correlation and Covariance
###############################################################################

    def _find_covariance(x, y):
        '''
        Uses the data from which x and y were generated to calculate
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
        '''
        Manually set the correlation between two quantities

        Given a correlation factor, the covariance and correlation
        between two variables is added to both objects.
        '''
        if factor > 1 or factor < -1:
            raise ValueError('Correlation factor must be between -1 and 1.')

        x = self
        ro_xy = factor
        sigma_xy = ro_xy*x.std*y.std

        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

    def set_covariance(self, y, sigma_xy):
        '''
        Manually set the covariance between two quantities

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
        '''
        Returns the covariance of the object and a specified variable.

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
        '''
        Returns the correlation factor of two measurements.

        Using the covariance, or finding the covariance if not defined,
        the correlation factor of two measurements is returned.
        '''
        x = self
        if y.name in x.covariance:
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
        '''
        Renames an object, requires a string.
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
                ExperimentalValue.error_method + ' method.\n'
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
                ExperimentalValue.error_method + ' method.\n'
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

    def __add__(self, other):
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
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = Measurement_Array(len(other))
            for i in range(result.size):
                result[i]=op.operation_wrap(op.sub, self, other[i])
                #result.append(op.operation_wrap(op.sub, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other-self
        else:
            return op.operation_wrap(op.sub, other, self)

    def __truediv__(self, other):
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
    
    def __neg__(self):
        import qexpy.error_operations as op
        return op.neg(self)

    def __len__(self):
        return self.info['Data'].size

    def __eq__(self, other):
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
        if type(other) in ExperimentalValue.CONSTANT:
            return self.mean >= other
        else:
            try:
                other.type
            except AttributeError:
                raise TypeError
            else:
                return self.mean >= other.mean            
    def sqrt(x):
        return sqrt(x)
    
    def log(x):
        return log(x)

    def exp(x):
        return exp(x)

    def e(x):
        return exp(x)

    def sin(x):
        return sin(x)

    def cos(x):
        return cos(x)

    def tan(x):
        return tan(x)

    def csc(x):
        return csc(x)

    def sec(x):
        return sec(x)

    def cot(x):
        return cot(x)

    def asin(x):
        return asin(x)

    def acos(x):
        return acos(x)

    def atan(x):
        return atan(x)

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
    Subclass of measurements, specified by the user and treated as variables
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
    Subclass of objects, which are measurements created by operations or
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
    Subclass of measurement objects, not neccesarily specified by the user,
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
    def __new__(subtype, shape, dtype=Measurement, buffer=None, offset=0,
          strides=None, order=None, name = None, units = None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        
        if name is not None:
            obj.name = name
        else:
            obj.name = 'unnamed_arr%d' % (Measurement_Array.id_number)
            
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
        if obj is None: return
        self.units = getattr(obj, 'units', None)
        self.name = getattr(obj, 'name', None)
        
    def __array_wrap__(self, out_arr, context=None):
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)
 
    def get_means(self, method="der"):
        '''Returns a numpy array with the means of the measurements, as calculated
        by the method (der, MC, MinMax)'''
        if self.size == 0:
            return np.ndarray(0)
        
        means = np.ndarray(shape=self.shape)
        
        for index, item in np.ndenumerate(self):
            if item is not None:
                if method == "MinMax":
                    means[index] = item.MinMax[0]
                elif method == "MC":
                    means[index] = item.MC[0]
                else:
                    means[index] = item.mean
            else:
                means[index] = 0
        return means
    
    def get_stds(self, method="der"):
        '''Returns an array with the errors of the measurements, as calculated
        by the method (der, MC, MinMax)'''
        if self.size == 0:
            return np.ndarray(0)
        
        stds = np.ndarray(shape=self.shape)
        
        for index, item in np.ndenumerate(self):
            if item is not None:              
                if method == "MinMax":
                    stds[index] = item.MinMax[1]
                elif method == "MC":
                    stds[index] = item.MC[1]
                else:
                    stds[index] = item.std
            else:
                stds[index] = 0
        return stds
    
    def mean(self, method="der"):
        '''Return mean of the means of the measurements'''
        #overides numpy mean()
        nparr = self.get_means(method)
        self.mean = nparr.mean()
        return self.mean

    def get_mean(self, method="der"):
        '''Return mean of the means of the measurements'''
        return self.mean(method)
    
    def std(self, ddof=1, method="der"):
        '''Return standard deviation of the means of each measurement'''
        nparr = self.get_means(method)
        self.std = nparr.std(ddof=ddof)
        return self.std 
    
    def get_std(self, ddof=1, method="der"):
        '''Return standard deviation of the means of each measurement'''
        return self.std(method)

   
    def get_error_weighted_mean(self, method="der"):
        '''Return error weighted mean and error of the measurements, as a measurement'''
        
        means = self.get_means(method)
        stds = self.get_stds(method)
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
                   
        self.error_weighted_mean =  (0. if sumw2==0 else mean/sumw2)
        self.error_weighted_std =  (0. if sumw2==0 else np.sqrt(1./sumw2))
            
        return Measurement(self.error_weighted_mean, self.error_weighted_std)
    
    def get_units_str(self):
        '''Returns a string with the units.
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
        
    
    def set_errors(self, error):
        '''Set all of the errors on the data points - either to a constant value, or to an array of values'''
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
            print("error array must be same length as data")
    
    def __str__(self):
        theString=''
        for i in range(self.size):
            theString += self[i].__str__()
            if i != self.size-1:
                theString += ',\n'
        return theString
            
def MeasurementArray(data, error=None, name=None, units=None):
    '''Function to easily construct a Measurement_Array'''
    
    array = Measurement_Array(0, name=name, units=units)
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
        Latex, or the default style. Using ''')


def set_error_method(chosen_method):
    '''
    Choose the method of error propagation to be used. Enter a string.

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
        ExperimentalValue.error_method = "Monte Carlo"
    elif chosen_method in min_max_list:
        ExperimentalValue.error_method = "Min Max"
    elif chosen_method in derr_list:
        ExperimentalValue.error_method = "Derivative"
    else:
        print("Method not recognized, using derivative method.")
        ExperimentalValue.error_method = "Derivative"


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
    
    if ExperimentalValue.error_method == 'Derivative':
        [mean, std] = variable.der
    elif ExperimentalValue.error_method == 'Monte Carlo':
        [mean, std] = variable.MC
    elif ExperimentalValue.error_method == 'Min Max':
        [mean, std] = variable.MinMax

    if method is not None:
        if ExperimentalValue.error_method is 'Derivative':
            [mean, std] = variable.der
        elif ExperimentalValue.error_method is 'Monte Carlo':
            [mean, std] = variable.MC
        elif ExperimentalValue.error_method is 'Min Max':
            [mean, std] = variable.MinMax

    return (mean, std,)


def _tex_print(self, method=None):
    '''
    Creates string used by __str__ in a style useful for printing in Latex,
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
    '''
    Returns string used by __str__ as two numbers representing mean and error
    to either the first non-zero digit of error or to a specified number of
    significant figures.
    '''
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
    '''
    Returns string used by __str__ as two numbers representing mean and
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
    '''Creates a histogram of the inputted data using Bokeh.
    '''
    if type(title) is str:
        hist_title = title
    elif title is None:
        hist_title = 'Histogram'
    else:
        print('Histogram title must be a string.')
        hist_title = 'Histogram'

    mean, std = _variance(data)

    p1 = bp.figure(title=hist_title, tools="save",
                   background_fill_color="#FFFF")

    hist, edges = np.histogram(data, bins=50)

    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="#036564", line_color="#033649")

    p1.line([mean]*2, [0, hist.max()*1.05], line_color='red',
            line_dash='dashed')
    p1.line([mean-std]*2, [0, hist.max()*1.1], line_color='red',
            line_dash='dashed')
    p1.line([mean+std]*2, [0, hist.max()*1.1], line_color='red',
            line_dash='dashed')

    if output == 'inline':
        bp.output_notebook()
    elif output == 'file':
        bp.output_file(hist_title+' histogram.html', title=hist_title)
    else:
        print('''Output must be either "file" or "inline", using "file"
              by default.''')
        bp.output_file(hist_title+' histogram.html', title=hist_title)
    bp.show(p1)


def numerical_partial_derivative(func, var, *args):
    '''
    Returns the parital derivative of a dunction with respect to var.

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
    '''
    Returns the first order derivative of a function.
    '''
    return (function(point+dx)-function(point))/dx


def _variance(*args, ddof=1):
    '''
    Returns a tuple of the mean and standard deviation of a data array.

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
    from math import sqrt

    w = np.power(std, -2)
    w_mean = sum(np.multiply(w, mean))/sum(w)
    w_std = 1/sqrt(sum(w))
    return (w_mean, w_std)


def reset_variables():
    '''
    Resets the ID number, directories and methods to their original values.
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
