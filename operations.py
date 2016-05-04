from Uncertainties import measurement
from Uncertainties import normalize
CONSTANT = (int,float,)

def check_values(a,b):
    if isinstance(a,measurement.CONSTANT):
        a=normalize(a)
    if isinstance(b,measurement.CONSTANT):
        b=normalize(b)
    return [a,b]

def add(a,b):
    [a,b]=check_values(a,b)
    #Addition by error propogation formula
    if measurement.method=="Derivative":  
        mean = a.mean+b.mean
        std = a.std+b.std
        name=a.name+'+'+b.name
        result = measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Derivative method.\n"
        
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean = a.mean+b.mean
        std = a.std+b.std
        name=a.name+'+'+b.name
        result = measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Min-Max method.\n"
        
    #If method specification is bad, MC method is used
    else:
        plus=lambda V: V[0]+V[1]
        result=measurement.monte_carlo(plus,a,b)
        result.rename(a.name+'+'+b.name)
        result.info['Method']+="Errors propagated by Monte Carlo method.\n"
    return result;

def sub(a,b):
    [a,b]=check_values(a,b)    
    #Addition by error propogation formula
    if measurement.method=="Derivative":
        b=mul(b,-1)
        result=add(a,b)
        result.rename(a.name+'-'+b.name)
        result.info['Method']+="Errors propagated by Derivative method.\n"
    
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        result=add(a,mul(b,-1))
        result.rename(a.name+'-'+b.name)
        result.info['Method']+="Errors propagated by Min-Max method.\n"
    #Monte Carlo method
    else:
        minus=lambda V: V[0]-V[1]
        result=measurement.monte_carlo(minus,a,b)
        result.rename(a.name+'+'+b.name)
    return result

def mul(a,b):
    [a,b]=check_values(a,b)
    #By error propogation formula    
    if measurement.method=="Derivative":          
        mean = a.mean*b.mean
        std = (a.std**2*b.mean**2 +
            b.std**2*a.mean**2)**(1/2)
        name=a.name+'*'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Derivative method.\n"
        
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=a.mean*b.mean+a.std*b.std
        std=a.mean*b.std+b.mean*a.std
        name=a.name+'*'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Min-Max method.\n"
            
    #If method specification is bad, MC method is used
    else:
        plus=lambda a,b: a*b
        result=measurement.monte_carlo(plus,a,b)
        result.rename(a.name+'*'+b.name)
        result.info['Method']+="Errors propagated by Monte Carlo method.\n"
    return result;
    
def div(a,b):
    [a,b]=check_values(a,b)
    #By error propgation
    if measurement.method=="Derivative": 
        from math import log
        
        mean = a.mean/b.mean
        std = (a.std**2/b.mean**2+
            b.std**2*(a.mean*log(b.mean))**2)**(1/2)
        name=a.name+'*'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Derivative method.\n"
            
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=(b.mean*a.std+a.mean*b.std)/(b.mean**2*b.std**2)
        std=(a.mean*b.mean+a.std*b.std+2*a.mean*b.std+2*b.mean*a.std)
        name=a.name+'*'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Min-Max method.\n"
        
    #If method specification is bad, MC method is used
    else:
        divide=lambda a,b: a/b
        result=measurement.monte_carlo(divide,a,b)
        result.rename(a.name+'+'+b.name)
        result.info['Method']+="Errors propagated by Monte Carlo method\n"
    return result;

def power(a,b):
    [a,b]=check_values(a,b)
    #By error propagation
    if measurement.method=='Derivative':
        from math import log
        mean=a.mean**b.mean
        std=((b.mean*a.mean**(b.mean-1)*a.std)**2+
                 (a.mean**y.mean*log(a.mean)*b.std)**2)**(1/2)
        name=a.name+'**'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Error propagated by Derivative Method\n"
    elif measurement.method=='MinMax':
        if (b<0):
            max_val=(a.mean+a.std)**(b.mean-b.std)
            min_val=(a.mean-a.std)**(b.mean+b.std)
        elif(b>=0):
            max_val=(a.mean+a.std)**(b.mean+b.std)
            min_val=(a.mean-a.std)**(b.mean-b.std)
        mid_val=(max_val+min_val)/2
        error=(max_val-min_val)/2
        name=a.name+'**'+b.name
        result=measurement(name,mid_val,error)
        result.info['Method']+="Error propagated by Min-Max method.\n"
    else:
        exponent=lambda a,b: a**b
        result=measurement.monte_carlo(exponent,a,b)
        result.rename(a.name+'**'+b.name)
        result.info['Method']+="Errors propagated by Monte Carlo method\n"
    return result;
        
        
def sin(value):
    from math import sin
    from math import cos
    mean=sin(value.mean)
    std=abs(cos(value.mean)*value.std)
    name="cos("+value.name+")"
    result=measurement(name,mean,std)
    result.info += "Errors propagated by Derivative method.\n"
    #Must set correlation
    return result;
    
def cos(value):
    from math import sin
    from math import cos
    mean=cos(value.mean)
    std=abs(sin(value.mean)*value.std)
    name="sin("+value.name+")"
    result=measurement(name,mean,std)
    result.info += "Errors propagated by Derivative method.\n"
    #Must set correlation
    return result;
    
def exp(value):
    from math import exp
    mean=exp(value.mean)
    std=abs(value.mean*mean*value.std)
    name="exp("+value.name+")"
    result=measurement(name,mean,std)
    result.info += "Errors propagated by Derivative method.\n"
    #Must set correlation
    return result;

def e(value):
    measurement.exp(value)
    
def log(value):
    from math import log
    mean=log(value.mean)
    std=abs(value.std/value.mean)
    name="log("+value.name+")"
    result=measurement(name,mean,std)
    result.info += "Errors propagated by Derivative method.\n"
    #Must set correlation
    return result;
