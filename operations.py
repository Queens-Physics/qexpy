from uncertainties import Measurement
from uncertainties import Function
from uncertainties import Constant
CONSTANT = (int,float,)

def dev(*args,der=None):
    '''
    Returns the standard deviation of a function of N arguments.
    
    Using the tuple of variables, passed in each operation that composes a
    function, the standard deviation is calculated by the derivative error
    propagation formula, including the covariance factor between each pair
    of variables. The derivative dictionary of a funciton must be passes by
    the der argument.
    '''
    std=0
    roots=()
    for arg in args:
        for i in range(len(arg.root)):
            if arg.root[i] not in roots:
                roots+=(arg.root[i],)
    for root in roots:
        std+=(der[root]*Measurement.register[root].std)**2
    for i in range(len(roots)):
        for j in range(len(roots)-i-1):
            cov=Measurement.register[roots[i]]\
                    .get_covariance(Measurement.register[roots[j+1+i]])
            std+=2*der[roots[i]]*der[roots[j+1+i]]*cov
    std=std**(1/2)
    return std;
    
def check_values(*args):
    '''
    Checks that the arguments are measurement type, otherwise a measurement
    is returned.
    
    All returned values are of measurement type, if values need to be 
    converted, this is done by calling the normalize function, which
    outputs a measurement object with no standard deviation.
    '''
    val=()
    for arg in args:
        if type(arg) in CONSTANT:
            val+=(Constant(arg),)
        else:
            val+=(arg,)
    return val

def check_formula(operation,a,b=None,func_flag=False):
    '''
    Checks if quantity being calculated is already in memory
    
    Using the formula string created for each operation as a key, the
    register of previously calculated operations is checked. If the
    quantity does exist, the previously calculated object is returned.
    '''
    if func_flag is False:
        if a.info["Formula"]+operation+b.info["Formula"] in \
                Measurement.formula_register:
            ID = Measurement.formula_register[a.info["Formula"]+operation\
                    + b.info["Formula"]]
            return Measurement.register[ID]
    else:
        if operation+'('+a.info["Formula"]+')' in Measurement.formula_register:
            ID = Measurement.formula_register[operation+'('+a.info["Formula"]\
                    + ')']
            return Measurement.register[ID]

def add(a,b):
    '''
    Returns a measurement object that is the sum of two other measurements.
    
    The sum can be taken by multiple methods, specified by the measurement
    class variable measurement.method. The derivative of this new object is
    also specifed by applying the chain rule to the input and the 
    derivative of the inputs.
    '''
    a,b=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.first_der[key]+b.first_der[key]
    if check_formula('+',a,b) is not None:
        return check_formula('+',a,b)
    #Addition by error propogation formula
    if Measurement.method=="Derivative":  
        mean=a.mean+b.mean     
        std=dev(a,b,der=first_der)
        result=Function(mean,std)
        
    #Addition by Min-Max method
    elif Measurement.method=="Min Max":
        mean=a.mean+b.mean
        std=a.std+b.std
        result=Function(mean,std)
        
    #If method specification is bad, MC method is used
    else:
        plus=lambda x,y: x+y
        result=Measurement.monte_carlo(plus,a,b)
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.add(a.info["Data"],b.info["Data"])
    result.first_der.update(first_der)
    result._update_info('+',a,b)
    return result;

def sub(a,b):
    '''
    Returns a measurement object that is the subtraction of two measurements.
    '''
    a,b=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.first_der[key]-b.first_der[key] 
    if check_formula('-',a,b) is not None:
        return check_formula('-',a,b)
        
    #Addition by error propogation formula
    if Measurement.method=="Derivative":
        mean=a.mean-b.mean
        std=dev(a,b,der=first_der)
        result=Function(mean,std)
    
    #Addition by Min-Max method
    elif Measurement.method=="Min Max":
        result=add(a,-b)
        
    #Monte Carlo method
    else:
        minus=lambda x,y: x-y
        result=Measurement.monte_carlo(minus,a,b)
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.subtract(a.info["Data"],b.info["Data"])
    result.first_der.update(first_der)
    result._update_info('-',a,b)
    return result

def mul(a,b):
    a,b=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.mean*b.first_der[key]+b.mean*a.first_der[key]
    if check_formula('*',a,b) is not None:
        return check_formula('*',a,b)
        
    #By error propogation formula    
    if Measurement.method=="Derivative":          
        mean=a.mean*b.mean
        std=dev(a,b,der=first_der)
        result=Function(mean,std)
        
    #Addition by Min-Max method
    elif Measurement.method=="Min Max":
        mean=a.mean*b.mean+a.std*b.std
        std=a.mean*b.std+b.mean*a.std
        result=Function(mean,std)
            
    #If method specification is bad, MC method is used
    else:
        plus=lambda a,b: a*b
        result=Measurement.monte_carlo(plus,a,b)
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.multiply(a.info["Data"],b.info["Data"])
    result.first_der.update(first_der)
    result._update_info('*',a,b)
    return result;
    
def div(a,b):
    a,b=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=(a.first_der[key]*b.mean-b.first_der[key]*a.mean)\
                / b.mean**2
    if check_formula('/',a,b) is not None:
        return check_formula('/',a,b)
        
    #By error propgation
    if Measurement.method=="Derivative": 
        mean=a.mean/b.mean
        std=dev(a,b,der=first_der)
        result=Function(mean,std)
            
    #Addition by Min-Max method
    elif Measurement.method=="Min Max":
        mean=(b.mean*a.std+a.mean*b.std)/(b.mean**2*b.std**2)
        std=(a.mean*b.mean+a.std*b.std+2*a.mean*b.std+2*b.mean*a.std)
        result=Function(mean,std)
        
    #If method specification is bad, MC method is used
    else:
        divide=lambda a,b: a/b
        result=Measurement.monte_carlo(divide,a,b)
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.divide(a.info["Data"],b.info["Data"])
    result.first_der.update(first_der)
    result._update_info('/',a,b)
    return result;

def power(a,b):
    from math import log
    
    a,b=check_values(a,b)
    #Propagating derivative of arguments
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.mean**b.mean*(b.first_der[key]*log(abs(a.mean))
                + b.mean/a.mean*a.first_der[key])  
    if check_formula('**',a,b) is not None:
        return check_formula('**',a,b)
    
    #By derivative method
    if Measurement.method=="Derivative":
        mean=a.mean**b.mean
        std=dev(a,b,der=first_der)
        result=Function(mean,std)
        
    #By min-max method
    elif Measurement.method=='Min Max':
        if (b<0):
            max_val=(a.mean+a.std)**(b.mean-b.std)
            min_val=(a.mean-a.std)**(b.mean+b.std)
        elif(b>=0):
            max_val=(a.mean+a.std)**(b.mean+b.std)
            min_val=(a.mean-a.std)**(b.mean-b.std)
        mid_val=(max_val+min_val)/2
        err=(max_val-min_val)/2
        result=Function(mid_val,err)
    
    #By Monte Carlo method
    else:
        exponent=lambda a,b: a**b
        result=Measurement.monte_carlo(exponent,a,b)
    if a.info["Data"] is not None and b.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.power(a.info["Data"],b.info["Data"])
    result.first_der.update(first_der)   
    result._update_info('**',a,b)
    return result;
        
        
def sin(x):
    from math import sin, cos
    
    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=cos(x.mean)*x.first_der[key]
    if check_formula('sin',x,func_flag=True) is not None:
        return check_formula('sin',x,func_flag=True)
        
    #By derivative method
    if Measurement.method=='Derivative':
        mean=sin(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
        
    #By Monte Carlo method
    else:
        import numpy as np
        sine=lambda x: np.sin(x)
        result=Measurement.monte_carlo(sine,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.sin(x.info["Data"])
    result.first_der.update(first_der)
    result._update_info('sin',x,func_flag=1)    
    return result;
    
def cos(x):
    from math import sin, cos

    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=-sin(x.mean)*x.first_der[key]    
    if check_formula('cos',x,func_flag=True) is not None:
        return check_formula('cos',x,func_flag=True)
    
    #By derivative method
    if Measurement.method=='Derivative':        
        mean=cos(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #By Monte Carlo method
    else:
        import numpy as np
        cosine=lambda x: np.cos(x)
        result=Measurement.monte_carlo(cosine,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.cos(x.info["Data"])
    result.first_der.update(first_der)
    result._update_info('cos',x,func_flag=1)
    return result;

def tan(x):
    from math import tan, cos
    
    def Sec(x):
        return 1/cos(x)
        
    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=Sec(x.mean)**2*x.first_der[key]
    if check_formula('tan',x,func_flag=True) is not None:
        return check_formula('tan',x,func_flag=True)
    
    #Derivative method
    elif Measurement.method=='Derivative':  
        mean=tan(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #Min-Max method
    elif Measurement.method=='MinMax':  
        pass

    #Monte Carlo method
    elif Measurement.method=='Monte Carlo':  
        import numpy as np
        tangent=lambda x: np.tan(x)
        result=Measurement.monte_carlo(tangent,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.tan(x.info["Data"]) 
    result.first_der.update(first_der)
    result._update_info('tan',x,func_flag=1)
    return result;
    
def sec(x):
    from math import cos, tan
    
    def Csc(x):
        return 1/sin(x)
        
    def Sec(x):
        return 1/cos(x)
        
    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=Sec(x.mean)*tan(x.mean)*x.first_der[key]
    if check_formula('sec',x,func_flag=True) is not None:
        return check_formula('sec',x,func_flag=True)
    
    #Derivative method
    elif Measurement.method=='Derivative':  
        mean=Sec(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #Min-Max method
    elif Measurement.method=='MinMax':  
        pass

    #Monte Carlo method
    elif Measurement.method=='Monte Carlo':  
        import numpy as np
        secant=lambda x: np.divide(np.cos(x))
        result=Measurement.monte_carlo(secant,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.sec(x.info["Data"]) 
    result.first_der.update(first_der)
    result._update_info('sec',x,func_flag=1)
    return result;

def csc(x):
    from math import sin, tan
    
    def Cot(x):
        return 1/tan(x)
        
    def Csc(x):
        return 1/sin(x)
    
    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=-Cot(x.mean)*Csc(x.mean)*x.first_der[key]
    if check_formula('csc',x,func_flag=True) is not None:
        return check_formula('csc',x,func_flag=True)
    
    #Derivative method
    elif Measurement.method=='Derivative':  
        mean=Csc(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #Min-Max method
    elif Measurement.method=='MinMax':  
        pass

    #Monte Carlo method
    elif Measurement.method=='Monte Carlo':  
        import numpy as np
        cosecant=lambda x: np.divide(np.sin(x))
        result=Measurement.monte_carlo(cosecant,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.csc(x.info["Data"]) 
    result.first_der.update(first_der)
    result._update_info('csc',x,func_flag=1)
    return result;

def cot(x):
    from math import sin, tan

    def Cot(x):
        return 1/tan(x)
        
    def Csc(x):
        return 1/sin(x)
    
    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=-Csc(x.mean)**2*x.first_der[key]
    if check_formula('cot',x,func_flag=True) is not None:
        return check_formula('cot',x,func_flag=True)
    
    #Derivative method
    elif Measurement.method=='Derivative':  
        mean=Cot(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #Min-Max method
    elif Measurement.method=='MinMax':  
        pass

    #Monte Carlo method
    elif Measurement.method=='Monte Carlo':  
        import numpy as np
        cotan=lambda x: np.divide(np.tan(x))
        result=Measurement.monte_carlo(cotan,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.cot(x.info["Data"]) 
    result.first_der.update(first_der)
    result._update_info('cot',x,func_flag=1)
    return result;
    
def exp(x):
    from math import exp

    x,=check_values(x)
    first_der={}
    for key in x.first_der:
        first_der[key]=exp(x.mean)*x.first_der[key]     
    if check_formula('exp',x,func_flag=True) is not None:
        return check_formula('exp',x,func_flag=True)
    
    #By derivative method
    if Measurement.method=='Derivative':
        mean=exp(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #By min-max method
    elif Measurement.method=='Min Max':
        min_val=exp(x.mean-x.std)
        max_val=exp(x.mean+x.std)
        mid_val=(max_val+min_val)/x
        err=(max_val-min_val)/2
        result=Function(mid_val,err)
        
    #By Monte Carlo method
    else:
        import numpy as np
        euler=lambda x: np.exp(x)
        result=Measurement.monte_carlo(euler,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.exp(x.info["Data"]) 
    result.first_der.update(first_der)
    result._update_info('exp',x,func_flag=1)
    return result;

def e(value):
    Measurement.exp(value)
    
def log(x):
    from math import log

    x,=check_values(x) 
    first_der={}
    for key in x.first_der:
        first_der[key]=1/x.mean*x.first_der[key]         
    if check_formula('log',x,func_flag=True) is not None:
        return check_formula('log',x,func_flag=True)
        
    #By derivative method
    if Measurement.method=='Derivative':
        mean=log(x.mean)
        std=dev(x,der=first_der)
        result=Function(mean,std)
    
    #By Monte Carlo method
    else:
        import numpy as np
        nat_log=lambda x: np.log(x)
        result=Measurement.monte_carlo(nat_log,x)
    if x.info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.log(x.info["Data"])
    result.first_der.update(first_der)
    result._update_info('log',x,func_flag=1)    
    return result;
  

def operation_wrap(operation,*args,func_flag=False):
    '''
    Function wrapper to convert existing, constant functions into functions
    which can handle measurement objects and return an error propagated by
    derivative, min-max, or Monte Carlo method.
    '''
    #if func_flag is not False:
    #    from math import sin,cos,tan,exp,log,cot,csc,sec
    args=check_values(args)
    if args[1] is not None:
        args[0].check_der(args[1])
        args[1].check_der(args[0])
    df={}
    for key in args[0].first_der:
        df[key]=diff[operation]*Measurement.register[key].first_der
    if check_formula(op_string[operation],args,func_flag) is not None:
        return check_formula(op_string[operation],args,func_flag)   

    #Derivative Method
    if Measurement.method=="Derivative":
        mean=operation(args)
        std=dev(args,der=df)
        result=Measurement(mean,std)
        
    #Min Max Method
        
    #Monte Carlo Method
    else:
        result=Measurement.monte_carlo(operation,args)
    if args[0].info["Data"] is not None:
        import numpy
        result.info["Data"]=numpy.operation(args[0].info["Data"])   
    result.first_der.update(df)
    result._update_info(op_string[operation],*args,func_flag)

diff={sin:lambda x,key: cos(x.mean)*x.first_der[key],         
    cos:lambda x,key: -sin(x.mean)*x.first_der[key],
    tan:lambda x,key: sec(x.mean)**2*x.first_der[key],        
    sec:lambda x,key: tan(x)*sec(x)*x.first_der[key],
    csc:lambda x,key: -cot(x)*csc(x)*x.first_der[key],   
    cot:lambda x,key: -csc(x)**2*x.first_der[key], 
    exp:lambda x,key: exp(x)*x.first_der[key],           
    log:lambda x,key: 1/x*x.first_der[key],
    add:lambda a,b,key: a.first_der[key]+b.first_der[key],
    sub:lambda a,b,key: a.first_der[key]-b.first_der[key],
    mul:lambda a,b,key: a.first_der[key]*b.mean + b.first_der[key]*a.mean,
    div:lambda a,b,key: (a.first_der[key]*b.mean-b.first_der[key]*a.mean) \
                                                                / b.mean**2,
    power:lambda a,b,key: a.mean**b.mean*(b.first_der[key]*log(abs(a.mean))
                                            + b.mean/a.mean*a.first_der[key],)

}
      
op_string={sin:'sin',cos:'cos',tan:'tan',csc:'csc',sec:'sec',cot:'cot',
           exp:'exp',log:'log',add:'+',sub:'-',mul:'*',div:'/',power:'**',}