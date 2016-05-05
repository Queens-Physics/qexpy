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
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.first_der[key]+b.first_der[key]
        
    #Addition by error propogation formula
    if measurement.method=="Derivative":  
        mean=a.mean+b.mean     
        std=(a.std**2+b.std**2+a.get_correlation(b)\
                +b.get_correlation(a))**(1/2)
        result=measurement(mean,std)
        
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=a.mean+b.mean
        std=a.std+b.std
        result=measurement(mean,std)
        
    #If method specification is bad, MC method is used
    else:
        plus=lambda V: V[0]+V[1]
        result=measurement.monte_carlo(plus,a,b)
    
    result.first_der.update(first_der)
    result.update_info('+',a,b)
    return result;

def sub(a,b):
    [a,b]=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.first_der[key]-b.first_der[key] 
    
    #Addition by error propogation formula
    if measurement.method=="Derivative":
        mean=a.mean-b.mean
        std=(a.std**2+b.std**2-a.get_correlation(b)-\
                b.get_correlation(a))**(1/2)
        result=measurement(mean,std)
    
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        result=add(a,mul(b,-1))
        
    #Monte Carlo method
    else:
        minus=lambda V: V[0]-V[1]
        result=measurement.monte_carlo(minus,a,b)
    
    result.first_der.update(first_der)
    result.update_info('-',a,b)
    return result

def mul(a,b):
    [a,b]=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=a.mean*b.first_der[key]+b.mean*a.first_der[key]
    
    #By error propogation formula    
    if measurement.method=="Derivative":          
        mean=a.mean*b.mean
        std=(a.std**2*b.mean**2 +
            b.std**2*a.mean**2)**(1/2)
        result=measurement(mean,std)
        
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=a.mean*b.mean+a.std*b.std
        std=a.mean*b.std+b.mean*a.std
        result=measurement(mean,std)
            
    #If method specification is bad, MC method is used
    else:
        plus=lambda a,b: a*b
        result=measurement.monte_carlo(plus,a,b)
    
    result.first_der.update(first_der)
    result.update_info('*',a,b)
    return result;
    
def div(a,b):
    [a,b]=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    for key in a.first_der:
        first_der[key]=(a.first_der[key]*b.mean-b.first_der[key]*a.mean)\
                / b.mean**2
        
    #By error propgation
    if measurement.method=="Derivative": 
        from math import log
        
        mean=a.mean/b.mean
        std=(a.std**2/b.mean**2+
            b.std**2*(a.mean*log(b.mean))**2)**(1/2)
        result=measurement(mean,std)
            
    #Addition by Min-Max method
    elif measurement.method=="MinMax":
        mean=(b.mean*a.std+a.mean*b.std)/(b.mean**2*b.std**2)
        std=(a.mean*b.mean+a.std*b.std+2*a.mean*b.std+2*b.mean*a.std)
        result=measurement(mean,std)
        
    #If method specification is bad, MC method is used
    else:
        divide=lambda a,b: a/b
        result=measurement.monte_carlo(divide,a,b)
    
    result.first_der.update(first_der)
    result.update_info('/',a,b)
    return result;

def power(a,b):
    [a,b]=check_values(a,b)
    #Propagating derivative of arguments    
    first_der={}
    a.check_der(b)
    b.check_der(a)
    #for key in a.first_der:
        #first_der[key]=
                
    #By error propagation
    if measurement.method=='Derivative':
        from math import log
        mean=a.mean**b.mean
        std=((b.mean*a.mean**(b.mean-1)*a.std)**2+
                 (a.mean*b.mean*log(a.mean)*b.std)**2)**(1/2)
        result=measurement(mean,std)
        
    elif measurement.method=='MinMax':
        if (b<0):
            max_val=(a.mean+a.std)**(b.mean-b.std)
            min_val=(a.mean-a.std)**(b.mean+b.std)
        elif(b>=0):
            max_val=(a.mean+a.std)**(b.mean+b.std)
            min_val=(a.mean-a.std)**(b.mean-b.std)
        mid_val=(max_val+min_val)/2
        error=(max_val-min_val)/2
        result=measurement(mid_val,error)
        
    else:
        exponent=lambda a,b: a**b
        result=measurement.monte_carlo(exponent,a,b)

    result.first_der.update(first_der)   
    result.update_info('**',a,b)
    return result;
        
        
def sin(x):
    from math import sin
    from math import cos
    
    first_der={}
    for key in x.first_der:
        first_der[key]=cos(x.mean)*x.first_der[key]
    
    if measurement.method=='Derivative':
        mean=sin(x.mean)
        std=abs(cos(x.mean)*x.std)
        result=measurement(mean,std)
        
    else:
        sine=lambda x: sin(x)
        result=measurement.monte_carlo(sine,x)

    result.first_der.update(first_der)
    result.update_info('sin',x,func_flag=1)    
    return result;
    
def cos(x):
    from math import sin
    from math import cos

    first_der={}
    for key in x.first_der:
        first_der[key]=-sin(x.mean)*x.first_der[key]    
    
    if measurement.method=='Derivative':        
        mean=cos(x.mean)
        std=abs(sin(x.mean)*x.std)
        result=measurement(mean,std)
    else:
        cosine=lambda x: cos(x)
        result=measurement.monte_carlo(cosine,x)
    
    result.first_der.update(first_der)
    result.update_info('cos',x,func_flag=1)
    return result;
    
def exp(x):
    from math import exp
    
    first_der={}
    for key in x.first_der:
        first_der[key]=exp(x.mean)*x.first_der[key]        
    
    if measurement.method=='Derivative':
        mean=exp(x.mean)
        std=abs(x.mean*mean*x.std)
        result=measurement(mean,std)
        
    elif measurement.method=='MinMax':
        min_val=exp(x.mean-x.std)
        max_val=exp(x.mean+x.std)
        mid_val=(max_val+min_val)/x
        error=(max_val-min_val)/2
        result=measurement(mid_val,error)

    else:
        euler=lambda x: exp(x)
        result=measurement.monte_carlo(euler,x)

    result.first_der.update(first_der)
    result.update_info('exp',x,func_flag=1)
    return result;

def e(value):
    measurement.exp(value)
    
def log(x):
    from math import log
    if measurement.method=='Derivative':
        mean=log(x.mean)
        std=abs(x.std/x.mean)
        result=measurement(mean,std)
        result.update_info('log',x,func_flag=1)
    else:
        nat_log=lambda x: log(x)
        result=measurement.monte_carlo(nat_log,x)
        result.update_info('log',x,func_flag=1)    
    return result;
