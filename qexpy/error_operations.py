import numpy as np
import math as m
import qexpy.utils as qu
from qexpy.error import Measurement_Array, Measurement, Constant, Function
import scipy.stats as st

CONSTANT = qu.number_types 
ARRAY = qu.array_types +(Measurement_Array,)
MEASUREMENT = (Measurement, Constant, Function)

###############################################################################
# Mathematical operations
###############################################################################


def neg(x):
    '''Returns the negitive of a Measurement object
    '''
    import qexpy.error as e

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
    '''Returns a Measurement object that is the sum of two other Measurements.

    The sum can be taken by multiple methods,  specified by the measurement
    class variable measurement.error_method. The derivative of this new object
    is also specifed by applying the chain rule to the input and the
    derivative of the inputs.
    '''
    if type(a) in ARRAY or type(b) in ARRAY:
        import numpy as np
        return np.add(a, b)

    elif type(a) in CONSTANT:
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
    '''Returns a Measurement object that is the subtraction of two Measurements.
    '''
    if type(a) in ARRAY or type(b) in ARRAY:
        import numpy as np
        return np.subtract(a, b)

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
    '''Returns the product of two values with propagated errors.
    '''
    if type(a) in ARRAY or type(b) in ARRAY:
        import numpy as np
        return np.multiply(a, b)

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
    '''Returns the quotient of two values with propagated errors.
    '''
    if type(a) in ARRAY or type(b) in ARRAY:
        import numpy as np
        return np.divide(a, b)

    if type(a) in CONSTANT:
        if type(b) in CONSTANT:
            return a if b ==0 else a/b
        else:
            return a if b ==0 else a/b.mean

    else:
        if type(b) in CONSTANT:
            return a.mean if b ==0 else a.mean/b
        else:
            return a.mean if b.mean ==0 else a.mean/b.mean


def power(a, b):
    '''Returns the power of two values with propagated errors.
    '''
    if type(a) in ARRAY or type(b) in ARRAY:
        import numpy as np
        return np.power(a, b)

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


###############################################################################
# Mathematical Functions
###############################################################################

def sqrt(x):
    '''Returns the square root of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.sqrt(x)
    elif type(x) in ARRAY:
        return np.sqrt(x)
    elif isinstance(x,MEASUREMENT):
        return m.sqrt(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))
    
def sin(x):
    '''Returns the sine of a Measurement with propagated errors.
    '''
    import math as m
    if type(x) in CONSTANT:
        return m.sin(x)
    elif type(x) in ARRAY:
        return np.sin(x)
    elif isinstance(x,MEASUREMENT):
        return m.sin(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def asin(x):
    '''Returns the arctangent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.asin(x)   
    elif type(x) in ARRAY:
        return np.arcsin(x)
    elif isinstance(x,MEASUREMENT):
        return m.asin(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def cos(x):
    '''Returns the cosine of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.cos(x)
    elif type(x) in ARRAY:
        return np.cos(x)
    elif isinstance(x,MEASUREMENT):
        return m.cos(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def acos(x):
    '''Returns the arctangent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.acos(x)
    elif type(x) in ARRAY:
        return np.arccos(x)
    elif isinstance(x,MEASUREMENT):
        return m.acos(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def tan(x):
    '''Returns the tangent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.tan(x)
    elif type(x) in ARRAY:
        return np.tan(x)
    elif isinstance(x,MEASUREMENT):
        return m.tan(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def atan(x):
    '''Returns the arctangent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.atan(x)
    elif type(x) in ARRAY:
        return np.arctan(x)
    elif isinstance(x,MEASUREMENT):
        return m.atan(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def sec(x):
    '''Returns the secant of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return 0. if m.cos(x) ==0 else 1./m.cos(x)
    elif type(x) in ARRAY:
        return 1./np.cos(x)
    elif isinstance(x,MEASUREMENT):
        return 0. if m.cos(x.mean) ==0 else 1./m.cos(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def csc(x):
    '''Returns the cosecant of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return 0. if m.sin(x) ==0 else 1./m.sin(x)
    elif type(x) in ARRAY:
        return 1./np.sin(x)
    elif isinstance(x,MEASUREMENT):
        return 0. if m.sin(x.mean) ==0 else 1./m.sin(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def cot(x):
    '''Returns the cotangent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return 0. if m.tan(x) ==0 else 1./m.tan(x)
    elif type(x) in ARRAY:
        return 1./np.tan(x)
    elif isinstance(x,MEASUREMENT):
        return 0. if m.tan(x.mean) ==0 else 1./m.tan(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def exp(x):
    '''Returns the exponent of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.exp(x)
    elif type(x) in ARRAY:
        return np.exp(x)
    elif isinstance(x,MEASUREMENT):
        return m.exp(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

def log(x):
    '''Returns the natural logarithm of a Measurement with propagated errors.
    '''
    import math as m

    if type(x) in CONSTANT:
        return m.log(x)
    elif type(x) in ARRAY:
        return np.log(x)
    elif isinstance(x,MEASUREMENT):
        return m.log(x.mean)
    else:
        raise TypeError("Unsupported type: "+str(type(x)))

###############################################################################
# Error Propagation Methods
###############################################################################


def find_minmax(function, *args):
    '''
    e.Function to use Min-Max method to find the best estimate value
    and error on a given function
    '''
    import numpy as np
    N=Measurement.minmax_n
    
    include = {csc: lambda x: x%m.pi != 0,
               sec: lambda x: x%m.pi != m.pi/2,
               cot: lambda x: x%m.pi != 0,
               div: lambda x: x != 0,
               log: lambda x: x >= 0,
               sqrt: lambda x: x >= 0,
               acos: lambda x: (x <= 1) & (x >= -1),
               asin: lambda x: (x <= 1) & (x >= -1)}

    if len(args) is 1:
        x = args[0]
        #vals = np.linspace(x.mean-x.std, x.mean + x.std, N)
        vals = np.linspace(x.MinMax[0]-x.MinMax[1], x.MinMax[0] + x.MinMax[1], N)
        if function in include:
            vals = vals[include[function](vals)]
        results = function(vals)
            
    elif len(args) is 2:     
        a = args[0]
        b = args[1]

        if function == power:
            a_vals = np.ndarray([])
            b_vals = np.ndarray([])
            for i in range(m.ceil(a.MinMax[0]-a.MinMax[1]), m.floor(a.MinMax[0]+a.MinMax[1])):
                np.append(a_vals, i)
            for i in range(m.ceil(b.MinMax[0]-b.MinMax[1]), m.floor(b.MinMax[0]+b.MinMax[1])):
                np.append(b_vals, i)
            if a.MinMax[0]-a.MinMax[1] > 0:
                a_vals = np.append(a_vals, np.linspace(a.MinMax[0]-a.MinMax[1], a.MinMax[0] + a.MinMax[1], N))
            elif a.MinMax[0]+a.MinMax[1] > 0:
                a_vals = np.append(a_vals, np.linspace(0, a.MinMax[0]+a.MinMax[1], N))   
            if b.MinMax[0]-b.MinMax[1] > 0:
                b_vals = np.append(b_vals, np.linspace(b.MinMax[0]-b.MinMax[1], b.MinMax[0] + b.MinMax[1], N))
            elif b.MinMax[0]+b.MinMax[1] > 0:
                b_vals = np.append(b_vals, np.linspace(0, b.MinMax[0]+b.MinMax[1], N))
        else:
            a_vals = np.linspace(a.MinMax[0]-a.MinMax[1], a.MinMax[0] + a.MinMax[1], N)
            b_vals = np.linspace(b.MinMax[0]-b.MinMax[1], b.MinMax[0] + b.MinMax[1], N)

        if function == div:
            b_vals = b_vals[include[function](b_vals)]

        results = np.ndarray(shape=(len(a_vals),len(b_vals)))
        
        for i in range(len(a_vals)):
            for j in range(len(b_vals)):
                results[i][j]= function(a_vals[i], b_vals[j])
    else:
        print("unsupported number of parameters")
        results = np.ndarray(0)
        
    
    min_val = results.min()
    max_val = results.max()
    mid_val = (max_val + min_val)/2.
    err = (max_val-min_val)/2.
    return [mid_val, err]


def monte_carlo(func, *args):
    '''Uses a Monte Carlo simulation to determine the mean and standard
    deviation of a function.

    Inputted arguments must be measurement type. Constants can be used
    as 'normalized' quantities which produce a constant row in the
    matrix of normally randomized values.
    '''
    # 2D array
    import numpy as np
    import qexpy.error as e

    _np_func = {add: np.add, sub: np.subtract, mul: np.multiply,
                div: np.divide, power: np.power, log: np.log,
                exp: np.exp, sin: np.sin, cos: np.cos, sqrt: np.sqrt,
                tan: np.tan, atan: np.arctan,
                csc: lambda x: np.divide(1, np.sin(x)),
                sec: lambda x: np.divide(1, np.cos(x)),
                cot: lambda x: np.divide(1, np.tan(x)),
                asin: np.arcsin, acos: np.arccos, atan: np.arctan,
                }

    # Domains of functions that have undefined points in their domain.
    # Returns True if the function is undefined at x.
    exclude = {csc: lambda x: x%m.pi == 0,
               sec: lambda x: x%m.pi == m.pi/2,
               cot: lambda x: x%m.pi == 0,
               div: lambda x: x == 0,
               log: lambda x: x < 0,
               sqrt: lambda x: x < 0,
               power: lambda x: x < 0,
               acos: lambda x: (x > 1) | (x < -1),
               asin: lambda x: (x > 1) | (x < -1)}

    N = len(args)
    n = e.ExperimentalValue.mc_trial_number
    value = np.zeros((N, n))
    result = np.zeros(n)
    rand = np.zeros((N, n))

    if(func == power and args[0].mean<0 and args[1].std!=0):
        return([np.nan, np.nan], result)
    if func in exclude:
        rand[0], x = _generate_excluded_normal(args[0].mean, args[0].std, n, exclude[func])
    elif args[0].MC_list is not None:
        value[0] = args[0].MC_list
    elif args[0].std == 0:
        value[0] = args[0].mean
        args[0].MC_list = value[0]
    else:
        rand[0] = np.random.normal(size=n) 

    if len(args) == 2:
        rand[1] = np.random.normal(size=n)
        args[0].get_covariance(args[1])
        rho = round(args[0]._get_correlation(args[1]), 8)
        factor = rho*rand[0] + np.sqrt(1-rho*rho)*rand[1] # rand[0] stores the excluded normal distribution
        if func == div:
            value[0] = factor*args[0].std+args[0].mean
            value[1] = rand[0]*args[1].std+args[1].mean
        else:
            value[1] = factor*args[1].std+args[1].mean
            if np.all(value[0]==0):
                value[0] = rand[0]*args[0].std+args[0].mean
    elif np.all(value[0]==0):
        value[0] = rand[0]*args[0].std+args[0].mean

    result = _np_func[func](*value)
    data = np.mean(result)
    error = np.std(result, ddof=1)
    return ([data, error], result,)

def _generate_excluded_normal(mean, sigma, n, excluded):
    '''Recursively generates random normal variables that are undefined 
    on some part of their codomain. Returns the normal distribution and
    the underlying standard normal distribution that was used to create it.
    '''
    standard = np.random.normal(size=n) # mean=0, std=1 normal distribution
    x = standard*sigma+mean             # mean=mean, std=std random normal distribution
    if np.any(excluded(x)):
        standard[excluded(x)], x[excluded(x)] = _generate_excluded_normal(mean, sigma, len(x[excluded(x)]), excluded)
    return standard, x

###############################################################################
# Methods for Propagation
###############################################################################


def operation_wrap(operation, *args, func_flag=False):
    '''e.Function wrapper to convert existing,  constant functions into functions
    which can handle measurement objects and return an error propagated by
    derivative,  min-max,  or Monte Carlo method.
    '''
    import qexpy.error as e

    args = check_values(*args)

    if len(args) > 1:
        args[0].get_covariance(args[1])

    if func_flag is False:
        args[0]._check_der(args[1])
        args[1]._check_der(args[0])

    df = {}
    for key in args[0].derivative:
        df[key] = diff[operation](key, *args)

    if check_formula(operation, *args, func_flag=func_flag) is not None:
        return check_formula(op_string[operation], *args, func_flag=func_flag)

    mean = operation(*args)
    std = dev(*args, der=df)
    result = e.Function(mean, std)
    result.der = [mean, std]
    result.MinMax = find_minmax(operation, *args)
    result.MC, result.MC_list = monte_carlo(operation, *args)

    #TODO: This is wrong: should not keep changing the mean and std
    # the error method should only change this on a print!!!
    
    # Derivative Method
    #if e.ExperimentalValue.error_method == "Derivative":
    #    pass

    # By Min-Max method
    #elif e.ExperimentalValue.error_method == "Min Max":
    #    (mean, std, ) = result.MinMax

    # Monte Carlo Method
    #elif e.ExperimentalValue.error_method == 'Monte Carlo':
    #    (mean, std, ) = result.MC

    #else:
    #    print('''Error method not properly set, please set to derivatie, Monte
    #    Carlo, or Min-Max. Derivative method used by default.''')

    if func_flag is False and args[0].info["Data"] is not None\
            and args[1].info['Data'] is not None\
            and args[0].info['Data'].size == args[1].info['Data'].size:
        d1 = args[0].info["Data"]
        d2 = args[1].info["Data"]
        result.info['Data'] = np.ndarray(d1.size, dtype=type(d1[0]))
        for i in range(d1.size):           
            result.info["Data"][i] = operation(d1[i],d2[i])

    elif args[0].info["Data"] is not None and func_flag is True:
        result.info["Data"] = (operation(args[0].info["Data"]))

    result.derivative.update(df)
    result._update_info(operation, *args, func_flag=func_flag)

    return result

diff = {
        sqrt: lambda key, x: (0. if x.mean ==0 else 0.5/m.sqrt(x.mean)*x.derivative[key]),
        sin: lambda key, x: m.cos(x.mean)*x.derivative[key],
        cos: lambda key, x: -m.sin(x.mean)*x.derivative[key],
        tan: lambda key, x: m.cos(x.mean)**-2*x.derivative[key],
        sec: lambda key, x: m.tan(x.mean)*m.cos(x.mean)**-1*x.derivative[key],
        csc: lambda key, x: -(m.tan(x.mean)*m.sin(x.mean))**-1 *
        x.derivative[key],

        cot: lambda key, x: -m.sin(x.mean)**-2*x.derivative[key],
        exp: lambda key, x: m.exp(x.mean)*x.derivative[key],
        log: lambda key, x: (0. if x.mean ==0 else 1./x.mean*x.derivative[key]),
        add: lambda key, a, b: a.derivative[key] + b.derivative[key],
        sub: lambda key, a, b: a.derivative[key] - b.derivative[key],
        mul: lambda key, a, b: a.derivative[key]*b.mean +
        b.derivative[key]*a.mean,

        div: lambda key, a, b: (a.derivative[key]*b.mean -
        b.derivative[key]*a.mean) / (1. if b.mean==0 else b.mean**2),

        power: lambda key, a, b: a.mean**b.mean*(
        b.derivative[key]*( 0. if a.mean<=0. else m.log(a.mean) ) +
        b.mean/(1. if a.mean ==0 else a.mean)*a.derivative[key]),

        asin: lambda key, x: (1-x.mean**2)**(-1/2)*x.derivative[key],
        acos: lambda key, x: -(1-x.mean**2)**(-1/2)*x.derivative[key],
        atan: lambda key, x: 1/(1 + x.mean**2)*x.derivative[key],
        }

op_string = {sin: 'sin', cos: 'cos', tan: 'tan', csc: 'csc', sec: 'sec', sqrt: 'sqrt',
             cot: 'cot', exp: 'exp', log: 'log', add: '+', sub: '-',
             mul: '*', div: '/', power: '**', asin: 'asin', acos: 'acos',
             atan: 'atan', }


#def dev(*args, der=None, manual_args=None):
def dev(*args, der=None):    
    '''Returns the standard deviation of a function of N arguments.

    Using the tuple of variables,  passed in each operations that composes a
    function,  the standard deviation is calculated by the derivative error
    propagation formula,  including the covariance factor between each pair
    of variables. The derivative dictionary of a function must be passes by
    the der argument.
    '''
    import qexpy.error as e


    std = 0
    roots = ()

    for arg in args:
        for i in range(len(arg.root)):
            if arg.root[i] not in roots:
                roots += (arg.root[i], )

    # Partial derivative times uncertainty terms
    for root in roots:
        std += (der[root]*e.ExperimentalValue.register[root].std)**2

    # Covariance terms
    for i in range(len(roots)):
        for j in range(len(roots)-i-1):
            cov = e.ExperimentalValue.register[roots[i]].get_covariance(
                e.ExperimentalValue.register[roots[j + 1 + i]])
            std += 2*der[roots[i]]*der[roots[j + 1 + i]]*cov
    std = std**(1/2)

    return std



def check_values(*args):
    '''Checks that the arguments are measurement type,  otherwise a measurement
    is returned.

    All returned values are of measurement type,  if values need to be
    converted,  this is done by calling the normalize function,  which
    outputs a measurement object with no standard deviation.
    '''
    import qexpy.error as e

    val = ()
    for arg in args:
        if type(arg) in CONSTANT:
            val += (e.Constant(arg), )
        else:
            val += (arg, )
    return val


def check_formula(operation, a, b=None, func_flag=False):
    '''Checks if quantity being calculated is already in memory

    Using the formula string created for each operation as a key,  the
    register of previously calculated operations is checked. If the
    quantity does exist,  the previously calculated object is returned.
    '''
    import qexpy.error as e

    op_string = {
        sin: 'sin', cos: 'cos', tan: 'tan', csc: 'csc', sec: 'sec', sqrt: 'sqrt',
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
