class ExperimentalValue:
    '''
    Root class of objects which containt a mean and standard deviation.
    From this class, objects with properties pertaining to their use or
    formulation can be instanced. (ie. the result of an operation of
    measured values, called Funciton and Measured respectivly)
    '''
    error_method = "Default"  # Default error propogation method
    default_style = "Default"  # Default printing style
    mc_trial_number = 10000  # number of trial in Monte Carlo simulation
    figs = None
    register = {}
    formula_register = {}
    id_register = {}

    # Defining common types under single arrayclear
    from numpy import int64, float64, ndarray, int32, float32
    CONSTANT = (int, float, int64, float64, int32, float32)
    ARRAY = (list, tuple, ndarray)
    try:
        import numpy
    except ImportError:
        print("Please install numpy for full features.")
        numpy_installed = False
    else:
        ARRAY += (numpy.ndarray,)
        numpy_installed = True

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

        elif all(isinstance(n, ExperimentalValue.ARRAY) for n in args):

            # Sample mean and std talen from list of values
            if len(args) == 1 and\
                    all(isinstance(n, ExperimentalValue.CONSTANT)
                        for n in args[0]):
                args = args[0]
                (self.mean, self.std) = variance(args)
                data = list(args)

            # List of Measurements, weighted mean and std taken
            elif len(args) == 1 and \
                    all(isinstance(n, ExperimentalValue) for n in args[0]):
                mean_vals = []
                std_vals = []
                for arg in args:
                    mean_vals.append(arg.mean)
                    std_vals.append(arg.std)
                (self.mean, self.std) = weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

            # Mean and std taken from list of values and associated errors
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
                (self.mean, self.std) = weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

        elif len(args) > 2:

            # Series of values entered, mean and std taken from said values
            if all(isinstance(n, ExperimentalValue.CONSTANT) for n in args):
                (self.mean, self.std) = variance(args)
                data = list(args)

            # Series of Measurements, weighted mean and std taken
            elif all(isinstance(n, ExperimentalValue) for n in args):
                mean_vals = []
                std_vals = []
                for arg in args:
                    mean_vals.append(arg.mean)
                    std_vals.append(arg.std)
                (self.mean, self.std) = weighted_variance(mean_vals, std_vals)
                data = mean_vals
                error_data = std_vals

        # Series of lists with mean and std in each, mean and std calculated
        elif all(len(arg) == 2 for arg in args):
            mean_vals = []
            std_vals = []
            for i in len(args):
                mean_vals.append(args[i][0])
                std_vals.append(args[i][1])
            (self.mean, self.std) = weighted_variance(mean_vals, std_vals)
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
        self.style = ExperimentalValue.default_style
        if name is not None:
            self.user_name = True
        else:
            self.user_name = False

        self.units = {}
        self.MC_list = None

    def set_method(chosen_method):
        '''
        Choose the method of error propagation to be used. Enter a string.

        Function to change default error propogation method used in
        measurement functions.
        '''
        mc_list = (
            'MC', 'mc', 'montecarlo', 'Monte Carlo', 'MonteCarlo',
            'monte carlo',)
        min_max_list = ('Min Max', 'MinMax', 'minmax', 'min max',)
        derr_list = ('Derivative', 'derivative', 'diff', 'der',)
        default_list = ('Default', 'default',)

        if chosen_method in mc_list:
            if ExperimentalValue.numpy_installed:
                ExperimentalValue.error_method = "Monte Carlo"
            else:
                ExperimentalValue.error_method = "Monte Carlo"
        elif chosen_method in min_max_list:
            ExperimentalValue.error_method = "Min Max"
        elif chosen_method in derr_list:
            ExperimentalValue.error_method = "Derivative"
        elif chosen_method in default_list:
            ExperimentalValue.error_method = "Default"
        else:
            print("Method not recognized, using default method.")
            ExperimentalValue.error_method = "Default"

    def __str__(self):
        '''
        Method called when printing measurement objects.
        '''
        if self.user_name:
            string = self.name+' = '
        else:
            string = ''

        if self.style == "Latex":
            string += tex_print(self)
        elif self.style == "Default":
            string += def_print(self)
        elif self.style == "Scientific":
            string += sci_print(self)
        try:
            self.error_flag
        except AttributeError:
            return string
        if self.error_flag is True:
            string = string + '\nErrors may be inacurate, '\
                + ' Monte Carlo method recommended.'
            return string
        else:
            return string

    def print_style(self, style, figs=None):
        '''
        Set the style of printing and number of significant figures for the
        output of a printing a measurement object.
        '''
        latex = ("Latex", "latex", 'Tex', 'tex',)
        Sci = ("Scientific", "Sci", 'scientific', 'sci', 'sigfigs',)
        ExperimentalValue.figs = figs

        if style in latex:
            self.style = "Latex"
        elif style in Sci:
            self.style = "Scientific"
        else:
            self.style = "Default"

    def MC_print(self):
        if ExperimentalValue.style == "Latex":
            string = tex_print(self, method=self.MC)
        elif ExperimentalValue.style == "Default":
            string = def_print(self, method=self.MC)
        elif ExperimentalValue.style == "Scientific":
            string = sci_print(self, method=self.MC)
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

    def set_correlation(self, y, factor):
        '''
        Manually set the correlation between two quantities

        Given a correlation factor, the covariance and correlation
        between two variables is added to both objects.
        '''
        x = self
        ro_xy = factor
        sigma_xy = ro_xy*x.std*y.std

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

    def _return_correlation(x, y):
        '''
        Returns the correlation factor of two measurements.

        Using the covariance, or finding the covariance if not defined,
        the correlation factor of two measurements is returned.
        '''
        if y.name in x.covariance:
            pass
        else:
            ExperimentalValue._find_covariance(x, y)

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

    def _update_info(self, operation, var1, var2=None, func_flag=None):
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

        op_string = {op.sin: 'sin', op.cos: 'cos', op.tan: 'tan',
                     op.csc: 'csc', op.sec: 'sec', op.cot: 'cot',
                     op.exp: 'exp', op.log: 'log', op.add: '+',
                     op.sub: '-', op.mul: '*', op.div: '/', op.power: '**',
                     'neg': '-', op.asin: 'asin', op.acos: 'acos',
                     op.atan: 'atan', }

        if func_flag is None and var2 is not None:
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

        elif func_flag is not None:
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

    def monte_carlo(func, *args):
        '''
        Uses a Monte Carlo simulation to determine the mean and standard
        deviation of a function.

        Inputted arguments must be measurement type. Constants can be used
        as 'normalized' quantities which produce a constant row in the
        matrix of normally randomized values.
        '''
        # 2D array
        import numpy as np
        import qexpy.error_operations as op

        _np_func = {op.add: np.add, op.sub: np.subtract, op.mul: np.multiply,
                    op.div: np.divide, op.power: np.power, op.log: np.log,
                    op.exp: np.exp, op.sin: np.sin, op.cos: np.cos,
                    op.tan: np.tan, op.atan: np.arctan,
                    op.csc: lambda x: np.divide(1, np.sin(x)),
                    op.sec: lambda x: np.divide(1, np.cos(x)),
                    op.cot: lambda x: np.divide(1, np.tan(x)),
                    op.asin: np.arcsin, op.acos: np.arccos, op.atan: np.arctan,
                    }

        N = len(args)
        n = ExperimentalValue.mc_trial_number
        value = np.zeros((N, n))
        result = np.zeros(n)
        for i in range(N):
            if args[i].MC_list is not None:
                value[i] = args[i].MC_list
            elif args[i].std == 0:
                value[i] = args[i].mean
                args[i].MC_list = value[i]
            else:
                value[i] = np.random.normal(args[i].mean, args[i].std, n)
                args[i].MC_list = value[i]

        result = _np_func[func](*value)
        data = np.mean(result)
        error = np.std(result, ddof=1)
        return (data, error,)


def set_default_print_style(style=None):
    if type(style) is str:
        ExperimentalValue.default_style(style)
    else:
        print('''A style must be a string of either: Scientific notation,
        Latex, or the default style.''')


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
        self.covariance = {self.name: self.std**2}
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
        self.covariance = {self.name: self.std**2}
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


def f(function, *args):
    '''
    Function wrapper for any defined function to operate with arbitrary
    measurement type objects arguments. Returns a Function type measurement
    object.
    '''
    N = len(args)
    mean = function(args)
    std_squared = 0
    for i in range(N):
        for arg in args:
            std_squared += arg.std**2*numerical_partial_derivative(
                function, i, args)**2
    std = (std_squared)**(1/2)
    argName = ""
    for i in range(N):
        argName += ', '+args[i].name
    name = function.__name__+"("+argName+")"
    return Function(mean, std, name=name)


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


def variance(*args, ddof=1):
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


def weighted_variance(mean, std, ddof=1):
    import numpy as np
    from math import sqrt

    w = np.power(std, -2)
    w_mean = sum(np.multiply(w, mean))/sum(w)
    w_std = 1/sqrt(sum(w))
    return (w_mean, w_std)


def tex_print(self, method=None):
    '''
    Creates string used by __str__ in a style useful for printing in Latex,
    as a value with error, in brackets multiplied by a power of ten. (ie.
    15+/-0.3 is (150 \pm 3)\e-1. Where Latex parses \pm as +\- and \e as
    *10**-1)
    '''
    if method is None:
        mean = self.mean
        std = self.std
    elif method == 'MC':
        [mean, std] = self.MC
    elif method == 'MinMax':
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


def def_print(self, method=None):
    '''
    Returns string used by __str__ as two numbers representing mean and error
    to either the first non-zero digit of error or to a specified number of
    significant figures.
    '''
    flag = True
    i = 0
    if method is None:
        mean = self.mean
        std = self.std
    elif method == 'MC':
        [mean, std] = self.MC
    elif method == 'MinMax':
        [mean, std] = self.MinMax

    if ExperimentalValue.figs is not None:
        value = abs(mean)
        while(flag):
            if value == 0:
                flag = False
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


def sci_print(self, method=None):
    '''
    Returns string used by __str__ as two numbers representing mean and
    error, each in scientific notation to a specified numebr of significant
    figures, or 3 if none is given.
    '''
    if method is None:
        mean = self.mean
        std = self.std
    elif method == 'MC':
        [mean, std] = self.MC
    elif method == 'MinMax':
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
    ExperimentalValue.default_style = "Default"
    ExperimentalValue.figs = 3
