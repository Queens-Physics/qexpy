from uncertainties import Measurement
from uncertainties import Function
from uncertainties import Constant
from numpy import int64, float64
CONSTANT = (int, float, int64, float64, )


def dev(*args, der=None):
    '''
    Returns the standard deviation of a function of N arguments.

    Using the tuple of variables,  passed in each operation that composes a
    function,  the standard deviation is calculated by the derivative error
    propagation formula,  including the covariance factor between each pair
    of variables. The derivative dictionary of a function must be passes by
    the der argument.
    '''
    std = 0
    roots = ()
    for arg in args:
        for i in range(len(arg.root)):
            if arg.root[i] not in roots:
                roots += (arg.root[i], )
    for root in roots:
        std += (der[root]*Measurement.register[root].std)**2
    for i in range(len(roots)):
        for j in range(len(roots)-i-1):
            cov = Measurement.register[roots[i]]\
                    .return_covariance(Measurement.register[roots[j + 1 + i]])
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
    val = ()
    for arg in args:
        if type(arg) in CONSTANT:
            val += (Constant(arg), )
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
    op_string = {
        sin: 'sin', cos: 'cos', tan: 'tan', csc: 'csc', sec: 'sec',
        cot: 'cot', exp: 'exp', log: 'log', add: '+', sub: '-',
        mul: '*', div: '/', power: '**', }
    op = op_string[operation]

    if func_flag is False:
        if a.info["Formula"] + op + b.info["Formula"] in \
                Measurement.formula_register:
            ID = Measurement.formula_register[a.info["Formula"] + op\
                     +  b.info["Formula"]]
            return Measurement.register[ID]
    else:
        if op + '(' + a.info["Formula"] + ')' in Measurement.formula_register:
            ID = Measurement.formula_register[op + '(' + a.info["Formula"]\
                     +  ')']
            return Measurement.register[ID]


def add(a, b):
    '''
    Returns a measurement object that is the sum of two other measurements.

    The sum can be taken by multiple methods,  specified by the measurement
    class variable measurement.method. The derivative of this new object is
    also specifed by applying the chain rule to the input and the
    derivative of the inputs.
    '''
    a, b = check_values(a, b)
    # Propagating derivative of arguments    
    first_der = {}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key] = diff[add](key, a, b)
    if check_formula(add, a, b) is not None:
        return check_formula(add, a, b)
    # Addition by error propogation formula
    if Measurement.method == "Derivative":
        mean = a.mean + b.mean     
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        
    # Addition by Min-Max method
    elif Measurement.method == "Min Max":
        mean = a.mean + b.mean
        std = a.std + b.std
        result = Function(mean, std)
        
    # If method specification is bad,  MC method is used
    elif Measurement.method == "Monte Carlo":
        (mean, std, ) = Measurement.monte_carlo(lambda x, y: x + y, a, b)
        result = Function(mean, std)
        
    else:
        mean = a.mean + b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x, y: x + y, a, b)
        result.MC = [MC[0], MC[1]]
        result.MinMax = [a.mean + b.mean, a.std + b.std]
        
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.add(a.info["Data"], b.info["Data"])
    
    result.unit = a.units    
    result.first_der.update(first_der)
    result._update_info(add, a, b)
    return result

def sub(a, b):
    '''
    Returns a measurement object that is the subtraction of two measurements.
    '''
    a, b = check_values(a, b)
    # Propagating derivative of arguments    
    first_der = {}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key] = a.first_der[key]-b.first_der[key] 
    if check_formula(sub, a, b) is not None:
        return check_formula(sub, a, b)
        
    # Addition by error propogation formula
    if Measurement.method == "Derivative":
        mean = a.mean-b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
    
    # Addition by Min-Max method
    elif Measurement.method == "Min Max":
        result = add(a, -b)
        
    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        (mean, std, ) = Measurement.monte_carlo(lambda x, y: x-y, a, b)
        result = Function(mean, std)
        
    else:
        mean = a.mean-b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x, y: x-y, a, b)
        result.MC = [MC[0], MC[1]]
        result.MinMax = [a.mean-b.mean, dev(a, b, der = first_der)]
        
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.subtract(a.info["Data"], b.info["Data"])
    result.units = a.units    
    result.first_der.update(first_der)
    result._update_info(sub, a, b)
    return result

def mul(a, b):
    a, b = check_values(a, b)
    # Propagating derivative of arguments    
    first_der = {}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key] = a.mean*b.first_der[key] + b.mean*a.first_der[key]
    if check_formula(mul, a, b) is not None:
        return check_formula(mul, a, b)
        
    # By error propogation formula    
    if Measurement.method == "Derivative":
        mean = a.mean*b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        
    # Addition by Min-Max method
    elif Measurement.method == "Min Max":
        mean = a.mean*b.mean + a.std*b.std
        std = a.mean*b.std + b.mean*a.std
        result = Function(mean, std)
            
    # If method specification is bad,  MC method is used
    elif Measurement.method == 'Monte Carlo':
        (mean, std, ) = Measurement.monte_carlo(lambda a, b: a*b, a, b)
        result = Function(mean, std)
    
    else:
        mean = a.mean*b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda a, b: a*b, a, b)
        result.MC = [MC[0], MC[1]]
        result.MinMax = [a.mean*b.mean + a.std*b.std, a.mean*b.std + b.mean*a.std]
      
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.multiply(a.info["Data"], b.info["Data"])
    
    units = a.units
    for key in units:
        if key in b.units:
            units[key] += b.units[key]
    result.units = units
    result.first_der.update(first_der)
    result._update_info(mul, a, b)
    return result
    
def div(a, b):
    a, b = check_values(a, b)
    # Propagating derivative of arguments    
    first_der = {}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key] = (a.first_der[key]*b.mean-b.first_der[key]*a.mean)\
                / b.mean**2
    if check_formula(div, a, b) is not None:
        return check_formula(div, a, b)
        
    # By error propgation
    if Measurement.method == "Derivative": 
        mean = a.mean/b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
            
    # Addition by Min-Max method
    elif Measurement.method == "Min Max":
        mean = (b.mean*a.std + a.mean*b.std)/(b.mean**2*b.std**2)
        std = (a.mean*b.mean + a.std*b.std + 2*a.mean*b.std + 2*b.mean*a.std)
        result = Function(mean, std)
        
    # If method specification is bad,  MC method is used
    elif Measurement.method == 'Monte Carlo':
        (mean, std, ) = Measurement.monte_carlo(lambda a, b: a/b, a, b)
        result = Function(mean, std)
        
    else:
        mean = a.mean/b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda a, b: a/b, a, b)
        result.MC = [MC[0], MC[1]]
        mean = (b.mean*a.std + a.mean*b.std)/(b.mean**2*b.std**2)
        std = (a.mean*b.mean + a.std*b.std + 2*a.mean*b.std + 2*b.mean*a.std)
        result.MinMax = [mean, std]
        
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.divide(a.info["Data"], b.info["Data"])
    
    units = a.units        
    for key in units:
        if key in b.units:
            units[key] -= b.units[key]
    result.units = units    
    result.first_der.update(first_der)
    result._update_info(div, a, b)
    return result

def power(a, b):
    import math as m    
    
    a, b = check_values(a, b)
    # Propagating derivative of arguments
    first_der = {}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key] = a.mean**b.mean*(b.first_der[key]*m.log(abs(a.mean))
                 +  b.mean/a.mean*a.first_der[key])  
    if check_formula(power, a, b) is not None:
        return check_formula(power, a, b)
    
    # By derivative method
    if Measurement.method == "Derivative":
        mean = a.mean**b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        
    # By min-max method
    elif Measurement.method == 'Min Max':
        if (b.mean<0):
            max_val = (a.mean + a.std)**(b.mean-b.std)
            min_val = (a.mean-a.std)**(b.mean + b.std)
        elif(b.mean >= 0):
            max_val = (a.mean + a.std)**(b.mean + b.std)
            min_val = (a.mean-a.std)**(b.mean-b.std)
        mid_val = (max_val + min_val)/2
        err = (max_val-min_val)/2
        result = Function(mid_val, err)
    
    # By Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        (mean, std, ) = Measurement.monte_carlo(lambda a, b: a**b, a, b)
        result = Function(mean, std)
    
    else:
        mean = a.mean**b.mean
        std = dev(a, b, der = first_der)
        result = Function(mean, std)
        if (b.mean<0):
            max_val = (a.mean + a.std)**(b.mean-b.std)
            min_val = (a.mean-a.std)**(b.mean + b.std)
        elif(b.mean >= 0):
            max_val = (a.mean + a.std)**(b.mean + b.std)
            min_val = (a.mean-a.std)**(b.mean-b.std)
        mid_val = (max_val + min_val)/2
        err = (max_val-min_val)/2
        result.MinMax = [mid_val, err]
        MC = Measurement.monte_carlo(lambda a, b: a**b, a, b)
        result.MC = [MC[0], MC[1]]
        
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.power(a.info["Data"], b.info["Data"])
     
    units = a.units
    for key in units:
        units[key] *= b.mean
    result.units = units 
    result.first_der.update(first_der)   
    result._update_info(power, a, b)
    return result
        
def sin(x):
    import math as m
    
    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = m.cos(x.mean)*x.first_der[key]
    if check_formula(sin, x, func_flag = True) is not None:
        return check_formula(sin, x, func_flag = True)
        
    # By derivative method
    if Measurement.method == 'Derivative':
        mean = m.sin(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)

    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: m.sin(x), x)
        
    # By Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.sin(x), x)
        result = Function(mean, std)
        
    else:
        import numpy as np
        mean = m.sin(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)   
        MC = Measurement.monte_carlo(lambda x: np.sin(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: m.sin(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.sin(x.info["Data"])
        
    result.first_der.update(first_der)
    result._update_info(sin, x, func_flag = 1)
    result.error_flag = True
    return result
    
def cos(x):
    import math as m

    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = -m.sin(x.mean)*x.first_der[key]    
    if check_formula(cos, x, func_flag = True) is not None:
        return check_formula(cos, x, func_flag = True)
    
    # By derivative method
    if Measurement.method == 'Derivative':        
        mean = m.cos(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)

    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: m.cos(x), x)
    
    # By Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.cos(x), x)
        result = Function(mean, std)
        
    else:
        import numpy as np
        mean = m.cos(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.cos(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: m.cos(x), x)
        result.MinMax = [MM.mean, MM.std]        
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.cos(x.info["Data"])
        
    result.first_der.update(first_der)
    result._update_info(cos, x, func_flag = 1)
    result.error_flag = True
    return result

def tan(x):
    import math as m
    
    def Sec(x):
        return 1/m.cos(x)
        
    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = Sec(x.mean)**2*x.first_der[key]
    if check_formula(tan, x, func_flag = True) is not None:
        return check_formula(tan, x, func_flag = True)
    
    # Derivative method
    elif Measurement.method == 'Derivative':  
        mean = m.tan(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: m.tan(x), x)

    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':  
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.tan(x), x)
        result = Function(mean, std)
    
    else:
        import numpy as np
        mean = m.tan(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.tan(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: m.tan(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.tan(x.info["Data"]) 
        
    result.first_der.update(first_der)
    result._update_info(tan, x, func_flag = 1)
    result.error_flag = True
    return result
    
def atan(x):
    x, = check_values(x)
    first_der = {}
    from math import atan
    for key in x.first_der:
        first_der[key] = 1/(1 + x.mean**2)*x.first_der[key]
    if check_formula(tan, x, func_flag = True) is not None:
        return check_formula(tan, x, func_flag = True)
    
    # Derivative method
    elif Measurement.method == 'Derivative':  
        mean = atan(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: atan(x), x)

    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':  
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.tan(x), x)
        result = Function(mean, std)
    
    else:
        import numpy as np
        mean = atan(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.tan(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: atan(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.tan(x.info["Data"]) 
        
    result.first_der.update(first_der)
    result._update_info(tan, x, func_flag = 1)
    result.error_flag = True
    return result
    
def sec(x):
    import math as m
    
    def Csc(x):
        return 1/m.sin(x)
        
    def Sec(x):
        return 1/m.cos(x)
        
    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = Sec(x.mean)*m.tan(x.mean)*x.first_der[key]
    if check_formula(sec, x, func_flag = True) is not None:
        return check_formula(sec, x, func_flag = True)
    
    # Derivative method
    elif Measurement.method == 'Derivative':  
        mean = Sec(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: Sec(x), x)

    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':  
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.divide(1, np.cos(x)), x)
        result = Function(mean, std)
        
    else:
        import numpy as np
        mean = Sec(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.divide(1, np.cos(x)), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: Sec(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.divide(numpy.cos(x.info["Data"]))
        
    result.first_der.update(first_der)
    result._update_info(sec, x, func_flag = 1)
    result.error_flag = True
    return result

def csc(x):
    import math as m
    
    def Cot(x):
        return 1/m.tan(x)
        
    def Csc(x):
        return 1/m.sin(x)
    
    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = -Cot(x.mean)*Csc(x.mean)*x.first_der[key]
    if check_formula(csc, x, func_flag = True) is not None:
        return check_formula(csc, x, func_flag = True)
    
    # Derivative method
    elif Measurement.method == 'Derivative':  
        mean = Csc(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: Csc(x), x)

    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':  
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(
                lambda x: np.divide(1, np.sin(x)), x)
        
    else:
        import numpy as np
        mean = Csc(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.divide(1, np.sin(x)), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: Csc(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.divide(numpy.sin(x.info["Data"]))
        result = Function(mean, std)
        
    result.first_der.update(first_der)
    result._update_info(csc, x, func_flag = 1)
    result.error_flag = True
    return result

def cot(x):
    import math as m

    def Cot(x):
        return 1/m.tan(x)
        
    def Csc(x):
        return 1/m.sin(x)
    
    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = -Csc(x.mean)**2*x.first_der[key]
    if check_formula(cot, x, func_flag = True) is not None:
        return check_formula(cot, x, func_flag = True)
    
    # Derivative method
    elif Measurement.method == 'Derivative':  
        mean = Cot(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: Cot(x), x)
        

    # Monte Carlo method
    elif Measurement.method == 'Monte Carlo':  
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(
                lambda x: np.divide(1, np.tan(x)), x)
        result = Function(mean, std)
    
    else:
        import numpy as np
        mean = Cot(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std) 
        MC = Measurement.monte_carlo(lambda x: np.divide(1, np.tan(x)), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: Cot(x), x)
        result.MinMax = [MM.mean, MM.std]
    
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.divide(numpy.tan(x.info["Data"]))
        
    result.first_der.update(first_der)
    result._update_info(cot, x, func_flag = 1)
    result.error_flag = True
    return result
    
def exp(x):
    import math as m

    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = m.exp(x.mean)*x.first_der[key]     
    if check_formula(exp, x, func_flag = True) is not None:
        return check_formula(exp, x, func_flag = True)
    
    # By derivative method
    if Measurement.method == 'Derivative':
        mean = m.exp(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: m.exp(x), x)
        
    # By Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.exp(x), x)
        result = Function(mean, std)
    
    else:
        import numpy as np
        mean = m.exp(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
        MC = Measurement.monte_carlo(lambda x: np.exp(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: m.exp(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.divide(numpy.exp(x.info["Data"])) 
        
    result.first_der.update(first_der)
    result._update_info(exp, x, func_flag = 1)
    return result

def e(value):
    Measurement.exp(value)

def log(x):
    import math as m

    x, = check_values(x)
    first_der = {}
    for key in x.first_der:
        first_der[key] = 1/x.mean*x.first_der[key]         
    if check_formula(log, x, func_flag = True) is not None:
        return check_formula(log, x, func_flag = True)
        
    # By derivative method
    if Measurement.method == 'Derivative':
        mean = m.log(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)
    
    # By Min-Max method
    if Measurement.method == "Min Max":
        return find_minmax(lambda x: m.log(x), x)
        
    # By Monte Carlo method
    elif Measurement.method == 'Monte Carlo':
        import numpy as np
        (mean, std, ) = Measurement.monte_carlo(lambda x: np.log(x), x)
        result = Function(mean, std)
        
    elif Measurement.method == 'Default':
        import numpy as np
        mean = m.log(x.mean)
        std = dev(x, der = first_der)
        result = Function(mean, std)        
        MC = Measurement.monte_carlo(lambda x: np.log(x), x)
        result.MC = [MC[0], MC[1]]
        MM = find_minmax(lambda x: m.log(x), x)
        result.MinMax = [MM.mean, MM.std]
        
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"] = numpy.log(x.info["Data"])
        
    result.first_der.update(first_der)
    result._update_info(log, x, func_flag = 1) 
    return result
  
def find_minmax(function, x):
    '''
    Function to use Min-Max method to find the best estimate value
    and error on a given function
    '''
    import numpy as np
    vals = np.linspace(x.mean-x.std, x.mean + x.std, 100)
    results = []
    for i in range(100):
        results.append(function(vals[i]))
    min_val = min(results)
    max_val = max(results)
    mid_val = (max_val + min_val)/2
    err = (max_val-min_val)/2
    return Function(mid_val, err)
    
def operation_wrap(operation, *args, func_flag = False):
    '''
    Function wrapper to convert existing,  constant functions into functions
    which can handle measurement objects and return an error propagated by
    derivative,  min-max,  or Monte Carlo method.
    '''
    args = check_values(args)
    if args[1] is not None:
        args[0].check_der(args[1])
        args[1].check_der(args[0])
    df = {}
    for key in args[0].first_der:
        df[key] = diff[operation](key, *args)
    if check_formula(op_string[operation], *args, func_flag) is not None:
        return check_formula(op_string[operation], *args, func_flag)   

    # Derivative Method
    if Measurement.method == "Derivative":
        mean = operation(*args)
        std = dev(*args, der = df)
        result = Function(mean, std)
        
    # By Min-Max method
    elif Measurement.method == "Min Max":
        return find_minmax(operation, *args)
        
    # Monte Carlo Method
    elif Measurement.method == 'Monte Carlo':
        (mean, std, ) = Measurement.monte_carlo(operation, *args)
    
    # Default method with all above method calculations
    else:
        mean = operation(*args)
        std = dev(*args, der = df)
        result = Function(mean, std)
        MM = find_minmax(operation, *args) 
        result.MinMax = [MM.mean, MM.std]
        MC = Measurement.monte_carlo(operation, *args)
        result.MC = [MC[0], MC[1]]
    
    if args[1] is not None and args[0].info["Data"] is not None\
                    and args[1].info['Data'] is not None\
                    and len(args[0]) == len(args[1]):
        for i in len(args[0]):
            result.info["Data"].append(
                        operation(args[0].info["Data"], args[1].info["Data"]))
                        
    elif args[0].info["Data"] is not None:
        result.info["Data"].append(operation(args[0].info["Data"]))
        
        
    result.first_der.update(df)
    result._update_info(op_string[operation], *args, func_flag)
    return result

diff = {sin:lambda key, x: cos(x.mean)*x.first_der[key],          
    cos:lambda key, x: -sin(x.mean)*x.first_der[key], 
    tan:lambda key, x: sec(x.mean)**2*x.first_der[key],         
    sec:lambda key, x: tan(x)*sec(x)*x.first_der[key], 
    csc:lambda key, x: -cot(x)*csc(x)*x.first_der[key],    
    cot:lambda key, x: -csc(x)**2*x.first_der[key],  
    exp:lambda key, x: exp(x)*x.first_der[key],            
    log:lambda key, x: 1/x*x.first_der[key], 
    add:lambda key, a, b: a.first_der[key] + b.first_der[key], 
    sub:lambda key, a, b: a.first_der[key]-b.first_der[key], 
    mul:lambda key, a, b: a.first_der[key]*b.mean  +  b.first_der[key]*a.mean, 
    div:lambda key, a, b: (a.first_der[key]*b.mean-b.first_der[key]*a.mean) \
                                                                / b.mean**2, 
    power:lambda key, a, b: a.mean**b.mean*(b.first_der[key]*log(abs(a.mean))
                                             +  b.mean/a.mean*a.first_der[key], )
}
      
op_string = {sin:'sin', cos:'cos', tan:'tan', csc:'csc', sec:'sec', cot:'cot', 
           exp:'exp', log:'log', add:'+', sub:'-', mul:'*', div:'/', power:'**', }