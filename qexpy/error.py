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
    figs = None
    register = {}
    formula_register = {}
    id_register = {}

    # Defining common types under single arrayclear
    from numpy import int64, float64, ndarray, int32, float32
    CONSTANT = (int, float, int64, float64, int32, float32)
    ARRAY = (list, tuple, ndarray)

    def __init__(self, *args, name=None):
        '''
        Creates a variable that contains a mean, standard deviation,
        and name for inputted data.
        '''
        data = None
        error_data = None

        # If two values eneterd first value is mean and second value is std
        if len(args) == 2 and all(
                isinstance(n, ExperimentalValue.CONSTANT)for n in args
                ):
            self.mean = args[0]
            self.std = args[1]

        # If an array and single value are entered, then error is uniform for
        # first array.
        elif len(args) == 2 and type(args[0]) in ExperimentalValue.ARRAY and\
                type(args[1]) in ExperimentalValue.CONSTANT:

            mean_vals = args[0]
            std_vals = [args[1]]*len(args[0])
            (self.mean, self.std) = _weighted_variance(mean_vals, std_vals)
            data = mean_vals
            error_data = std_vals

        # Checks that all arguments are array type
        elif all(isinstance(n, ExperimentalValue.ARRAY) for n in args):

            # Sample mean and std talen from single array of values
            if len(args) == 1 and\
                    all(isinstance(n, ExperimentalValue.CONSTANT)
                        for n in args[0]):
                args = args[0]
                (self.mean, self.std) = _variance(args)
                data = list(args)

            # Single array of Measurements, weighted mean and std taken
            elif len(args) == 1 and \
                    all(isinstance(n, ExperimentalValue) for n in args[0]):
                mean_vals = []
                std_vals = []
                for arg in args[0]:
                    mean_vals.append(arg.mean)
                    std_vals.append(arg.std)
                (self.mean, self.std) = _weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

            # Mean and std taken from array of values and array of errors
            elif len(args) == 2 and \
                    all(
                    isinstance(n, ExperimentalValue.CONSTANT)for n in args[0]
                    ) and\
                    all(
                    isinstance(n, ExperimentalValue.CONSTANT)for n in args[1]):
                mean_vals = args[0]
                if len(args[1]) == 1:
                    std_vals = args[1]*len(args[0])
                else:
                    std_vals = args[1]
                (self.mean, self.std) = _weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

            # For series of arrays of length 2, element 0 is mean, 1 is std.
            elif len(args) > 2 and all(len(n) is 2 for n in args):
                mean_vals = []
                std_vals = []
                for arg in args:
                    mean_vals.append(arg[0])
                    std_vals.append(arg[1])
                (self.mean, self.std) = _weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

            else:
                raise TypeError('''Measurement input must be either a single
                array of values, or two arrays with mean values and error
                values respectivly.''')

        elif len(args) > 2:
            # Series of values entered, mean and std taken from said values
            if all(isinstance(n, ExperimentalValue.CONSTANT) for n in args):
                (self.mean, self.std) = _variance(args)
                data = list(args)

            # Series of Measurements, weighted mean and std taken
            elif all(isinstance(n, ExperimentalValue) for n in args):
                mean_vals = []
                std_vals = []
                for arg in args:
                    mean_vals.append(arg.mean)
                    std_vals.append(arg.std)
                (self.mean, self.std) = _weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

        # If that fails, where than will you go
        else:
            raise ValueError('''Input arguments must be one of: a mean and
            standard deviation, an array of values, or the individual values
            themselves.''')

        self.info = {
                'ID': '', 'Formula': '', 'Method': '', 'Data': data,
                'Error': error_data, 'Function': {
                        'operation': (), 'variables': ()}, }

        ExperimentalValue.id_register[id(self)] = self
        self.print_style = ExperimentalValue.print_style
        if name is not None:
            self.user_name = True
        else:
            self.user_name = False

        self.units = {}
        self.MC_list = None

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

        unit_string = ''
        if self.units != {}:
            for key in self.units:
                if self.units[key] == 1 and len(self.units.keys()) is 1:
                    unit_string = key + unit_string
                else:
                    unit_string += key+'^%d' % (self.units[key])
                    unit_string += ' '
            unit_string = '['+unit_string+']'
            string += unit_string

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

    def print_min_mix_error(self):
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
            string = _def_print(self, method=self.MinMax)
        elif self.print_style == "Scientific":
            string = _sci_print(self, method=self.MinMax)
        print(string)

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
        sigma_xy = 0
        for i in range(len(data_x)):
            sigma_xy += (data_x[i]-x.mean)*(data_y[i]-y.mean)
        sigma_xy /= (len(data_x)-1)

        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

        factor = sigma_xy/x.std/y.std
        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

    def set_correlation(self, y, factor):
        '''
        Manually set the correlation between two quantities

        Given a correlation factor, the covariance and correlation
        between two variables is added to both objects.
        '''
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
        factor = sigma_xy/x.std/y.std

        x.correlation[y.info['ID']] = factor
        y.correlation[x.info['ID']] = factor

        x.covariance[y.info['ID']] = sigma_xy
        y.covariance[x.info['ID']] = sigma_xy

    def return_covariance(self, variable):
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

    def _return_correlation(self, y):
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
        return sigma_xy/sigma_x/sigma_y

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

        op_string = {op.sin: 'sin', op.cos: 'cos', op.tan: 'tan',
                     op.csc: 'csc', op.sec: 'sec', op.cot: 'cot',
                     op.exp: 'exp', op.log: 'log', op.add: '+',
                     op.sub: '-', op.mul: '*', op.div: '/', op.power: '**',
                     'neg': '-', op.asin: 'asin', op.acos: 'acos',
                     op.atan: 'atan', }

        if func_flag is False and var2 is not None:
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

        elif func_flag is True:
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
            print('Something went wrong in update_info')

    def return_derivative(self, variable=None):
        '''
        Returns the numerical value of the derivative with respect to an
        inputed variable.

        Function to find the derivative of a measurement or measurement like
        object. By default, derivative is with respect to itself, which will
        always yeild 1. Operator acts by acessing the self.derivative
        dictionary and returning the value stored there under the specific
        variable inputted (ie. deriving with respect to variable = ???)
        '''
        if variable is not None \
                and not hasattr(variable, 'type'):
            return 'Only measurement objects can be derived.'
        elif variable is None:
            return self.derivative
        if variable.info['ID'] not in self.derivative:
            self.derivative[variable.info['ID']] = 0
        derivative = self.derivative[variable.info["ID"]]
        return derivative

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

# Operations on measurement objects

    def __add__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.add, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other+self
        else:
            return op.operation_wrap(op.add, self, other)

    def __radd__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.add, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other+self
        else:
            return op.operation_wrap(op.add, self, other)

    def __mul__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.mul, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other*self
        else:
            return op.operation_wrap(op.mul, self, other)

    def __rmul__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.mul, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other*self
        else:
            return op.operation_wrap(op.mul, self, other)

    def __sub__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            print(other.mean)
            for value in other:
                result.append(op.operation_wrap(op.sub, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self-other
        else:
            return op.operation_wrap(op.sub, self, other)

    def __rsub__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.sub, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other-self
        else:
            return op.operation_wrap(op.sub, other, self)

    def __truediv__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.div, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self/other
        else:
            return op.operation_wrap(op.div, self, other)

    def __rtruediv__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.div, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other/self
        else:
            return op.operation_wrap(op.div, other, self)

    def __pow__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.power, self, value))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return self**other
        else:
            return op.operation_wrap(op.power, self, other)

    def __rpow__(self, other):
        import qexpy.error_operations as op
        if type(other) in ExperimentalValue.ARRAY:
            result = []
            for value in other:
                result.append(op.operation_wrap(op.power, value, self))
            return result
        elif type(self) in ExperimentalValue.CONSTANT and\
                type(other) in ExperimentalValue.CONSTANT:
            return other**self
        else:
            return op.operation_wrap(op.power, other, self)

    def sqrt(x):
        if x.mean < 0:
            print('Imaginary numbers are no supported in QExPy.')
        elif type(x) in ExperimentalValue.CONSTANT:
            import math as m
            return m.sqrt(x)
        else:
            import qexpy.error_operations as op
            return op.operation_wrap(op.power, x, 1/2)

    def __neg__(self):
        import qexpy.error_operations as op
        return op.neg(self)

    def __len__(self):
        return len(self.info['Data'])

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

    def log(x):
        return log(x)

    def show_MC_histogram(self, title=None):
        '''Creates and shows a Bokeh plot of a histogram of the values
        calculated by a Monte Carlo error propagation.
        '''
        import numpy as np
        from bokeh.plotting import figure, show, output_file

        if type(title) is str:
            hist_title = title
        elif title is None:
            hist_title = self.name+' Histogram'
        else:
            print('Histogram title must be a string.')
            hist_title = self.name+' Histogram'

        p1 = figure(title=hist_title, tools="save",
                    background_fill_color="#E8DDCB")

        hist, edges = np.histogram(self.MC_list, bins=50)

        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#036564", line_color="#033649")

        p1.line([self.mean]*2, [0, hist.max()*1.05], line_color='red',
                line_dash='dashed')
        p1.line([self.mean-self.std]*2, [0, hist.max()*1.1], line_color='red',
                line_dash='dashed')
        p1.line([self.mean+self.std]*2, [0, hist.max()*1.1], line_color='red',
                line_dash='dashed')

        output_file(self.name+' histogram.html', title=hist_title)
        show(p1)


def set_print_style(style=None, figs=None):
    '''Change style of printout for Measurement objects.

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
    ExperimentalValue.figs = figs

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
        self.MC = None
        self.MinMax = None
        self.error_flag = False


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


class Constant(ExperimentalValue):
    '''
    Subclass of measurement objects, not neccesarily specified by the user,
    called when a consant (int, float, etc.) is used in operation with a
    measurement. This class is called before calculating operations to
    ensure objects can be combined. The mean of a constant is the specified
    value, the standard deviation is zero, and the derivarive with respect
    to anything is zero.
    '''
    def __init__(self, arg):
        super().__init__(arg, 0)
        self.name = '%d' % (arg)
        self.info['ID'] = 'Constant'
        self.info["Formula"] = '%f' % arg
        self.derivative = {}
        self.info["Data"] = [arg]
        self.type = "Constant"
        self.covariance = {self.name: 0}
        self.root = ()


def sqrt(x):
    if x.mean < 0:
        print('Imaginary numbers are no supported in QExPy.')
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.sqrt(x)
    else:
        import qexpy.error_operations as op
        return op.operation_wrap(op.power, x, 1/2)


def sin(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.sin, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.sin(x)
    else:
        return op.operation_wrap(op.sin, x, func_flag=True)


def cos(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.cos, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.cos(x)
    else:
        return op.operation_wrap(op.cos, x, func_flag=True)


def tan(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.tan, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.tan(x)
    else:
        return op.operation_wrap(op.tan, x, func_flag=True)


def sec(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.sec, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.cos(x)
    else:
        return op.operation_wrap(op.sec, x, func_flag=True)


def csc(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.csc, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.sin(x)
    else:
        return op.operation_wrap(op.csc, x, func_flag=True)


def cot(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.cot, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return 1/m.tan(x)
    else:
        return op.operation_wrap(op.cot, x, func_flag=True)


def log(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.log, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.log(x)
    else:
        return op.operation_wrap(op.log, x, func_flag=True)


def exp(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.exp, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.exp(x)
    else:
        return op.operation_wrap(op.exp, x, func_flag=True)


def e(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.exp, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.exp(x)
    else:
        return op.operation_wrap(op.exp, x, func_flag=True)


def asin(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.asin, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.asin(x)
    else:
        return op.operation_wrap(op.asin, x, func_flag=True)


def acos(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.acos, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.acos(x)
    else:
        return op.operation_wrap(op.acos, x, func_flag=True)


def atan(x):
    import qexpy.error_operations as op
    if type(x) in ExperimentalValue.ARRAY:
        result = []
        for value in x:
            result.append(op.operation_wrap(op.atan, value, func_flag=True))
        return result
    elif type(x) in ExperimentalValue.CONSTANT:
        import math as m
        return m.atan(x)
    else:
        return op.operation_wrap(op.atan, x, func_flag=True)


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
    mean = sum(args)/len(args)
    for i in range(N):
        Sum += args[i]
        SumSq += args[i]*args[i]
    std = ((SumSq-Sum**2/N)/(N-1))**(1/2)
    return (mean, std)


def _weighted_variance(mean, std, ddof=1):
    import numpy as np
    from math import sqrt

    w = np.power(std, -2)
    w_mean = sum(np.multiply(w, mean))/sum(w)
    w_std = 1/sqrt(sum(w))
    return (w_mean, w_std)


def _tex_print(self, method=None):
    '''
    Creates string used by __str__ in a style useful for printing in Latex,
    as a value with error, in brackets multiplied by a power of ten. (ie.
    15+/-0.3 is (150 \pm 3)\e-1. Where Latex parses \pm as +\- and \e as
    *10**-1)
    '''
    if ExperimentalValue.error_method == 'Derivative':
        mean = self.mean
        std = self.std
    elif ExperimentalValue.error_method == 'Monte Carlo':
        [mean, std] = self.MC
    elif ExperimentalValue.error_method == 'Min Max':
        [mean, std] = self.MinMax

    if method is not None:
        if ExperimentalValue.error_method is 'Derivative':
            mean = self.mean
            std = self.std
        elif ExperimentalValue.error_method is 'Monte Carlo':
            [mean, std] = self.MC
        elif ExperimentalValue.error_method is 'Min Max':
            [mean, std] = self.MinMax

    flag = True
    i = 0

    if ExperimentalValue.figs is not None:
        value = abs(mean)
        while(flag):
            if value == 0:
                std = int(std/10**i//1)
                mean = int(mean/10**i//1)
                return "(%d \pm %d)\e%d" % (mean, std, i)
            elif value == float('inf'):
                return "inf"
            elif value < 1:
                value *= 10
                i -= 1
            elif value >= 10:
                value /= 10
                i += 1
            elif value >= 1 and value < 10:
                flag = False
        std = std*10**-i*10**(ExperimentalValue.figs-1)
        mean = mean*10**-i*10**(ExperimentalValue.figs-1)
        if i-ExperimentalValue.figs is not -1:
            return "(%d \pm %d)\e%d" % (mean, std,
                                        i-ExperimentalValue.figs + 1)
        else:
            return "(%d \pm %d)" % (mean, std)

    else:
        value = abs(std)
        while(flag):
            if value == 0:
                std = int(std)
                mean = int(mean)
                return "(%d \pm %d)\e%d" % (mean, std, i)
            elif value == float('inf'):
                return "%d \pm inf" % (mean)
            elif value < 1:
                value *= 10
                i -= 1
            elif value >= 10:
                value /= 10
                i += 1
            elif value >= 1 and value < 10:
                flag = False
        std = int(std/10**i)
        mean = int(mean/10**i)
        if i is not 0:
            return "(%d \pm %d)\e%d" % (mean, std, (i))
        else:
            return "(%d \pm %d)" % (mean, std)


def _def_print(self, method=None):
    '''
    Returns string used by __str__ as two numbers representing mean and error
    to either the first non-zero digit of error or to a specified number of
    significant figures.
    '''
    flag = True
    i = 0

    if ExperimentalValue.error_method == 'Derivative':
        mean = self.mean
        std = self.std
    elif ExperimentalValue.error_method == 'Monte Carlo':
        [mean, std] = self.MC
    elif ExperimentalValue.error_method == 'Min Max':
        [mean, std] = self.MinMax

    if method is not None:
        if ExperimentalValue.error_method == 'Derivative':
            mean = self.mean
            std = self.std
        elif ExperimentalValue.error_method == 'Monte Carlo':
            [mean, std] = self.MC
        elif ExperimentalValue.error_method == 'Min Max':
            [mean, std] = self.MinMax

    if ExperimentalValue.figs is not None:
        value = abs(mean)
        while(flag):
            if value == 0:
                flag = False
            elif value == float('inf'):
                return "inf"
            elif value < 1:
                value *= 10
                i += 1
            elif value >= 10:
                value /= 10
                i -= 1
            elif value >= 1 and value <= 10:
                flag = False
        figs = ExperimentalValue.figs+i-1
        if figs > 0:
            n = '%d' % (figs)
            n = "%."+n+"f"
        else:
            n = '%.0f'
        std = float(round(std, i))
        mean = float(round(mean, i))
        return n % (mean)+" +/- "+n % (std)

    else:
        value = abs(std)
        while(flag):
            if value == 0:
                flag = False
            elif value == float('inf'):
                return "%d \pm inf" % (mean)
            elif value < 1:
                value *= 10
                i += 1
            elif value >= 10:
                value /= 10
                i -= 1
            elif value >= 1 and value < 10:
                flag = False
        if i > 0:
            n = '%d' % (i)
            n = "%."+n+"f"
        else:
            n = '%.0f'
        std = float(round(std, i))
        mean = float(round(mean, i))
        return n % (mean)+" +/- "+n % (std)


def _sci_print(self, method=None):
    '''
    Returns string used by __str__ as two numbers representing mean and
    error, each in scientific notation to a specified numebr of significant
    figures, or 3 if none is given.
    '''
    if ExperimentalValue.error_method == 'Derivative':
        mean = self.mean
        std = self.std
    elif ExperimentalValue.error_method == 'Monte Carlo':
        [mean, std] = self.MC
    elif ExperimentalValue.error_method == 'Min Max':
        [mean, std] = self.MinMax

    if method is not None:
        if ExperimentalValue.error_method == 'Derivative':
            mean = self.mean
            std = self.std
        elif ExperimentalValue.error_method == 'Monte Carlo':
            [mean, std] = self.MC
        elif ExperimentalValue.error_method == 'Min Max':
            [mean, std] = self.MinMax

    flag = True
    i = 0

    if ExperimentalValue.figs is not None:
        value = abs(mean)
        while(flag):
            if value == 0:
                std = int(std/10**i//1)
                mean = int(mean/10**i//1)
                return "(%d \pm %d)\e%d" % (mean, std, i)
            elif value == float('inf'):
                return "inf"
            if value < 1:
                value *= 10
                i -= 1
            elif value >= 10:
                value /= 10
                i += 1
            elif value >= 1 and value < 10:
                flag = False
        std = std*10**-i*10**(ExperimentalValue.figs-1)
        mean = mean*10**-i*10**(ExperimentalValue.figs-1)
        if i-ExperimentalValue.figs is not -1:
            return "(%d +/- %d)*10**%d" % (round(mean), round(std),
                                           i-ExperimentalValue.figs + 1)
        else:
            return "(%d +/- %d)" % (round(mean), round(std))

    else:
        value = abs(std)
        while(flag):
            if value == 0:
                std = int(std)
                mean = int(mean)
                return "(%d \pm %d)\e%d" % (mean, std, i)
            elif value == float('inf'):
                return "%d \pm inf" % (mean)
            elif value < 1:
                value *= 10
                i -= 1
            elif value >= 10:
                value /= 10
                i += 1
            elif value >= 1 and value < 10:
                flag = False
        std = int(std/10**i)
        mean = int(mean/10**i)
        if i is not 0:
            return "(%d +/- %d)*10**%d" % (round(mean), round(std), (i))
        else:
            return "(%d +/- %d)" % (round(mean), round(std))


def Measurement_Array(data, error, name=None, units=None):
    ''' Creates an array of measurements from inputted mean and standard
    deviation arrays.
    '''
    import numpy as np

    if type(data) not in ExperimentalValue.ARRAY:
        print('Data array must be a list, tuple, or numpy array.')
        return None

    if type(error) not in ExperimentalValue.ARRAY:
        print('Error array must be a list, tuple, or numpy array.')
        return None

    if len(error) is 1:
        error = len(data)*[error[0]]

    if len(data) != len(error):
        print('''Data and error array must be of the same length, or the
        error array should be of length 1.''')
        return None

    data_name = name
    data_units = units

    measurement = []
    for i in range(len(data)):
        measurement.append(Measurement(data[i], error[i], name=data_name,
                                       units=data_units))

    return np.array(measurement)


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
