from numpy import int64, float64, ndarray
import math as m
CONSTANT = (int, float, int64, float64, )
ARRAY = (list, tuple, ndarray)


def dev(*args, der=None):
    '''
    Returns the standard deviation of a function of N arguments.

    Using the tuple of variables,  passed in each operation that composes a
    function,  the standard deviation is calculated by the derivative error
    propagation formula,  including the covariance factor between each pair
    of variables. The derivative dictionary of a function must be passes by
    the der argument.
    '''
    import error as e

    std = 0
    roots = ()

    for arg in args:
        for i in range(len(arg.root)):
            if arg.root[i] not in roots:
                roots += (arg.root[i], )
        for root in roots:
            std += (der[root]*e.ExperimentalValue.register[root].std)**2

    for i in range(len(roots)):
        for j in range(len(roots)-i-1):
            cov = e.ExperimentalValue.register[roots[i]].return_covariance(
                e.ExperimentalValue.register[roots[j + 1 + i]])
            std += 2*der[roots[i]]*der[roots[j + 1 + i]]*cov
    std = std**(1/2)

    return std


def check_values(*args):
    '''
    Checks that the arguments are measurement type,  otherwise a measurement
    is returned.

    All returned values are of measurement type,  if values need to be
    converted,  this is done by calling the normalize function,  which
    outputs a measurement object with no standard deviation.
    '''
    import error as e

    val = ()
    for arg in args:
        if type(arg) in CONSTANT:
            val += (e.Constant(arg), )
        else:
            val += (arg, )
    return val


def check_formula(operation, a, b=None, func_flag=False):
    '''
    Checks if quantity being calculated is already in memory

    Using the formula string created for each operation as a key,  the
    register of previously calculated operations is checked. If the
    quantity does exist,  the previously calculated object is returned.
    '''
    import error as e

    op_string = {
        sin: 'sin', cos: 'cos', tan: 'tan', csc: 'csc', sec: 'sec',
        cot: 'cot', exp: 'exp', log: 'log', add: '+', sub: '-',
        mul: '*', div: '/', power: '**', 'neg': '-', asin: 'asin',
        acos: 'acos', atan: 'atan', }

    op = op_string[operation]

    # check_formula is not behanving properly, requires overwrite, disabled
    return None

    if func_flag is False:
        if a.info["Formula"] + op + b.info["Formula"] in \
                e.ExperimentalValue.formula_register:
            ID = e.ExperimentalValue.formula_register[
                a.info["Formula"] + op + b.info["Formula"]]
            return e.ExperimentalValue.register[ID]

    else:
        if op + '(' + a.info["Formula"] + ')' in\
                    e.ExperimentalValue.formula_register:
            ID = e.ExperimentalValue.formula_register[
                op + '(' + a.info["Formula"] + ')']
            return e.ExperimentalValue.register[ID]


def neg(x):
    '''
    Returns the negitive of a measurement object
    '''
    import error as e

    x, = check_values(x)
    result_derivative = {}
    for key in x.derivative:
        result_derivative[key] = -x.derivative[key]
    result = e.Function(-x.mean, x.std)
    result.derivative.update(result_derivative)
    result._update_info('neg', x, func_flag=1)
    result.error_flag = True
    return result


def add(a, b):
    '''
    Returns a measurement object that is the sum of two other measurements.

    The sum can be taken by multiple methods,  specified by the measurement
    class variable measurement.error_method. The derivative of this new object
    is also specifed by applying the chain rule to the input and the
    derivative of the inputs.
    '''
    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a+b
        else:
            return a+b.mean

    elif type(a) and type(b) in ARRAY:
        result = []
        for i in range(len(a)):
            result.append(a[i] + b[i])
        return result

    else:
        if type(b) in CONSTANT:
            return a.mean+b
        else:
            return a.mean+b.mean


def sub(a, b):
    '''
    Returns a measurement object that is the subtraction of two measurements.
    '''
    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a-b
        else:
            return a-b.mean

    else:
        if type(b) in CONSTANT:
            return a.mean-b
        else:
            return a.mean-b.mean


def mul(a, b):
    '''Returns the product of two values with propagated errors.'''
    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a*b
        else:
            return a*b.mean

    else:
        if type(b) in CONSTANT:
            return a.mean*b
        else:
            return a.mean*b.mean


def div(a, b):
    '''Returns the quotient of two values with propagated errors.'''
    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a/b
        else:
            return a/b.mean

    else:
        if type(b) in CONSTANT:
            return a.mean/b
        else:
            return a.mean/b.mean


def power(a, b):
    '''Returns the power of two values with propagated errors.'''
    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a**b
        else:
            return a**b.mean

    else:
        if type(b) in CONSTANT:
            return a.mean**b
        else:
            return a.mean**b.mean


def sin(x):
    '''Returns the sine of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.sin(x)

    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.sin(x[i]))
        return result

    else:
        return m.sin(x.mean)


def asin(x):
    '''Returns the arctangent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.asin(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.asin(x[i]))
        return result
    else:
        return m.asin(x.mean)


def cos(x):
    '''Returns the cosine of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.cos(x)

    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.cos(x[i]))
        return result

    else:
        return m.cos(x.mean)


def acos(x):
    '''Returns the arctangent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.acos(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.acos(x[i]))
        return result
    else:
        return m.acos(x.mean)


def tan(x):
    '''Returns the tangent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.tan(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.tan(x[i]))
        return result
    else:
        return m.tan(x.mean)


def atan(x):
    '''Returns the arctangent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.atan(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.atan(x[i]))
        return result
    else:
        return m.atan(x.mean)


def sec(x):
    '''Returns the secant of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return 1/m.cos(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(1/m.cos(x[i]))
        return result
    else:
        return 1/m.cos(x.mean)


def csc(x):
    '''Returns the cosecant of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return 1/m.sin(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(1/m.sin(x[i]))
        return result
    else:
        return 1/m.sin(x.mean)


def cot(x):
    '''Returns the cotangent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return 1/m.tan(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(1/m.tan(x[i]))
        return result
    else:
        return 1/m.tan(x.mean)


def exp(x):
    '''Returns the exponent of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.exp(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.exp(x[i]))
        return result
    else:
        return m.exp(x.mean)


def log(x):
    '''Returns the natural logarithm of a measurement with propagated errors'''
    import math as m

    if type(x) in CONSTANT:
        return m.log(x)
    elif type(x) in ARRAY:
        result = []
        for i in range(len(x)):
            result.append(m.log(x[i]))
        return result
    else:
        return m.log(x.mean)


def find_minmax(function, *args):
    '''
    e.Function to use Min-Max method to find the best estimate value
    and error on a given function
    '''
    import numpy as np
    import error as e

    if len(args) is 1:
        x = args[0]
        vals = np.linspace(x.mean-x.std, x.mean + x.std, 100)
        results = []
        for i in range(100):
            results.append(function(vals[i]))

    elif len(args) is 2:
        a = args[0]
        b = args[1]
        results = []
        a_vals = np.linspace(a.mean-a.std, a.mean + a.std, 10)
        b_vals = np.linspace(b.mean-b.std, b.mean + b.std, 10)
        for i in range(10):
            results.append(function(a_vals[i], b_vals[i]))

    min_val = min(results)
    max_val = max(results)
    mid_val = (max_val + min_val)/2
    err = (max_val-min_val)/2
    return e.Function(mid_val, err)


def operation_wrap(operation, *args, func_flag=False):
    '''
    e.Function wrapper to convert existing,  constant functions into functions
    which can handle measurement objects and return an error propagated by
    derivative,  min-max,  or Monte Carlo method.
    '''
    import error as e

    args = check_values(*args)

    if func_flag is False:
        args[0]._check_der(args[1])
        args[1]._check_der(args[0])

    df = {}
    for key in args[0].derivative:
        df[key] = diff[operation](key, *args)

    if check_formula(operation, *args, func_flag=func_flag) is not None:
        return check_formula(op_string[operation], *args, func_flag=func_flag)

    # Derivative Method
    if e.ExperimentalValue.error_method == "Derivative":
        mean = operation(*args)
        std = dev(*args, der=df)
        result = e.Function(mean, std)

    # By Min-Max method
    elif e.ExperimentalValue.error_method == "Min Max":
        return find_minmax(operation, *args)

    # Monte Carlo Method
    elif e.ExperimentalValue.error_method == 'Monte Carlo':
        (mean, std, ) = e.ExperimentalValue.monte_carlo(operation, *args)

    # Default method with all above method calculations
    else:
        mean = operation(*args)
        std = dev(*args, der=df)
        result = e.Function(mean, std)
        MM = find_minmax(operation, *args)
        result.MinMax = [MM.mean, MM.std]
        MC = e.ExperimentalValue.monte_carlo(operation, *args)
        result.MC = [MC[0], MC[1]]

    if func_flag is False and args[0].info["Data"] is not None\
            and args[1].info['Data'] is not None\
            and len(args[0].info['Data']) == len(args[1].info['Data']):
        for i in range(len(args[0].info['Data'])):
            result.info["Data"].append(
                        operation(args[0].info["Data"][i],
                                  args[1].info["Data"][i]))

    elif args[0].info["Data"] is not None and func_flag is True:
        result.info["Data"].append(operation(args[0].info["Data"]))

    result.derivative.update(df)
    result._update_info(operation, *args, func_flag=func_flag)
    return result


diff = {sin: lambda key, x: m.cos(x.mean)*x.derivative[key],
        cos: lambda key, x: -m.sin(x.mean)*x.derivative[key],
        tan: lambda key, x: m.cos(x.mean)**-2*x.derivative[key],
        sec: lambda key, x: m.tan(x.mean)*m.cos(x.mean)**-1*x.derivative[key],
        csc: lambda key, x: -(m.tan(x.mean)*m.sin(x.mean))**-1 *
        x.derivative[key],

        cot: lambda key, x: -m.sin(x.mean)**-2*x.derivative[key],
        exp: lambda key, x: m.exp(x.mean)*x.derivative[key],
        log: lambda key, x: 1/x.mean*x.derivative[key],
        add: lambda key, a, b: a.derivative[key] + b.derivative[key],
        sub: lambda key, a, b: a.derivative[key] - b.derivative[key],
        mul: lambda key, a, b: a.derivative[key]*b.mean +
        b.derivative[key]*a.mean,

        div: lambda key, a, b: (a.derivative[key]*b.mean -
        b.derivative[key]*a.mean) / b.mean**2,

        power: lambda key, a, b: a.mean**b.mean*(
        b.derivative[key]*m.log(abs(a.mean)) +
        b.mean/a.mean*a.derivative[key]),

        asin: lambda key, x: (1-x.mean**2)**(-1/2)*x.derivative[key],
        acos: lambda key, x: -(1-x.mean**2)**(-1/2)*x.derivative[key],
        atan: lambda key, x: 1/(1 + x.mean**2)*x.derivative[key],
        }

op_string = {sin: 'sin', cos: 'cos', tan: 'tan', csc: 'csc', sec: 'sec',
             cot: 'cot', exp: 'exp', log: 'log', add: '+', sub: '-',
             mul: '*', div: '/', power: '**', asin: 'asin', acos: 'acos',
             atan: 'atan', }
