from Uncertainties import measurement
from Uncertainties import normalize
#from math import pi
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
        mean=a.mean+b.mean
        std=(a.std**2+b.std**2+a.get_correlation(b)+b.get_correlation(a))**(1/2)
        name=a.name+'+'+b.name
        result = measurement(mean,std)
        result.update_info(a,b,'+')
        
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
        mean=a.mean-b.mean
        std=(a.std**2+b.std**2-a.get_correlation(b)-b.get_correlation(a))**(1/2)
        name=a.name+'-'+b.name
        result = measurement(name,mean,std)
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
        result.rename(a.name+'-'+b.name)
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
        name=a.name+'/'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Derivative method.\n"
            
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=(b.mean*a.std+a.mean*b.std)/(b.mean**2*b.std**2)
        std=(a.mean*b.mean+a.std*b.std+2*a.mean*b.std+2*b.mean*a.std)
        name=a.name+'/'+b.name
        result=measurement(name,mean,std)
        result.info['Method']+="Errors propagated by Min-Max method.\n"
        
    #If method specification is bad, MC method is used
    else:
        divide=lambda a,b: a/b
        result=measurement.monte_carlo(divide,a,b)
        result.rename(a.name+'/'+b.name)
        result.info['Method']+="Errors propagated by Monte Carlo method\n"
    
    return result;

def power(a,b):
    [a,b]=check_values(a,b)
    #By error propagation
    if measurement.method=='Derivative':
        from math import log
        mean=a.mean**b.mean
        std=((b.mean*a.mean**(b.mean-1)*a.std)**2+
                 (a.mean*b.mean*log(a.mean)*b.std)**2)**(1/2)
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
        
        
def sin(x):
    from math import sin
    from math import cos
    if measurement.method=='Derivative':
        mean=sin(x.mean)
        std=abs(cos(x.mean)*x.std)
        name="cos("+x.name+")"
        result=measurement(name,mean,std)
        result.info["Method"]+="Errors propagated by Derivative method.\n"
    else:
        sine=lambda x: sin(x)
        result=measurement.monte_carlo(sine,x)
        result.rename('sin('+x.name+')')
        result.info['Method']+="Error propagated by Monte Carlo method.\n"
    
    return result;
    
def cos(x):
    from math import sin
    from math import cos
    if measurement.method=='Derivative':        
        mean=cos(x.mean)
        std=abs(sin(x.mean)*x.std)
        name="cos("+x.name+")"
        result=measurement(name,mean,std)
        result.info["Method"]+="Errors propagated by Derivative method.\n"
    else:
        cosine=lambda x: cos(x)
        result=measurement.monte_carlo(cosine,x)
        result.rename('sin('+x.name+')')
        result.info['Method']+="Error proagated by Monte Carlo method.\n"
    return result;
    
def exp(x):
    from math import exp
    if measurement.method=='Derivative':
        mean=exp(x.mean)
        std=abs(x.mean*mean*x.std)
        name="exp("+x.name+")"
        result=measurement(name,mean,std)
        result.info["Method"]+="Errors propagated by Derivative method.\n"
    elif measurement.method=='MinMax':
        min_val=exp(x.mean-x.std)
        max_val=exp(x.mean+x.std)
        mid_val=(max_val+min_val)/x
        error=(max_val-min_val)/2
        name="exp("+x.name+")"
        result=measurement(name,mid_val,error)
        result.info['Method']+="Error proagated by Min Max method.\n"
    else:
        euler=lambda x: exp(x)
        result=measurement.monte_carlo(euler,x)
        result.rename('exp('+x.name+')')
        result.info['Method']+="Error propagated by Monte Carlo method.\n"
    
    return result;

def e(value):
    measurement.exp(value)
    
def log(x):
    from math import log
    if measurement.method=='Derivative':
        mean=log(x.mean)
        std=abs(x.std/x.mean)
        name="log("+x.name+")"
        result=measurement(name,mean,std)
        result.info["Method"]+="Errors propagated by Derivative method.\n"
    else:
        nat_log=lambda x: log(x)
        result=measurement.monte_carlo(nat_log,x)
        result.rename('log('+x.name+')')
        result.info['Method']+="Error propagated by Monte Carlo mehtod.\n"
    
    return result;


