class measurement:
    method="Derivative" #Default error propogation method
    mcTrials=10000 #option for number of trial in Monte Carlo simulation
    id_number=1    
    
    #Defining common types under single array
    CONSTANT = (int,float,)
    ARRAY = (list,)
    try:
        import numpy
    except ImportError:
        pass
    else:
        ARRAY+=(numpy.ndarray,)
    
    def __init__(self,name,*args):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for inputted data.
        '''
        import numpy
        if len(args)==2 and type(args)==measurement.CONSTANT:
            self.mean=args[0]
            self.std=args[1]
            
        elif len(args)>2 or type(args[0])==measurement.ARRAY and len(args)==1:
            self.mean = numpy.mean(args)
            self.std = numpy.std(args,ddof=1)
        else:
            raise ValueError('''Input arguments must be one of: a mean and 
            standard deviation, an array of values, or the individual values
            themselves.''')
        self.name=name
        self.correlation={'Name': [name], 
            'Correlation Factor': [1]}
        self.info=""
        self.ID="var%d"%(measurement.id_number)
        measurement.id_number+=1
    
    def set_method(aMethod):
        '''
        Function to change default error propogation method used in measurement
        functions.
        '''
        if aMethod=="Monte Carlo":
            measurement.method="MonteCarlo"
        elif aMethod=="Min Max":
            measurement.method="MinMax"
        else:
            measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def set_correlation(self,variable,correlation):
        '''
        Adds a correlation factor between the variable being acted on and 
        any specified variable. Correlation factor must be specified.
        '''
        self.correlation['Name'].append(variable.name)
        self.correlation['Correlation Factor'].append(correlation)
        
        variable.correlation['Name'].append(self.name)
        variable.correlation['Correlation Factor'].append(correlation)
    
    def rename(self,newName):
        self.name=newName
    
###########################################################################
#Operations on measurement objects    
    
    def __add__(self,other):
        #Addition by error propogation formula
        if measurement.method=="Derivative":
            norm_other=norm_check(self,other)
            
            mean = self.mean+norm_other.mean
            std = self.std+norm_other.std
            name=self.name+'+'+norm_other.name
            result = measurement(name,mean,std)
            result.info+="Errors propgated by Derivative method"
            
        #Addition by Min-Max method
        elif measurement.method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]+V[1]
                result=measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other, measurement.CONSTANT):
                plus=lambda V: V[0]+other
                result=measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'+ {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        return result;
    def __radd__(self,other):
        return measurement.__add__(self,other);  

    def __mul__(self,other):
        if measurement.method=="Derivative":          
            norm_other=norm_check(self,other)
            
            mean = self.mean*norm_other.mean
            std = (self.std**2*norm_other.mean**2 +
                norm_other.std**2*self.mean**2)**(1/2)
            name=self.name+'*'+norm_other.name
            result = measurement(name,mean,std)
            result.info+="Errors propgated by Derivative method"
            
         #Addition by Min-Max method
        elif measurement.method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]*V[1]
                result=measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other, measurement.CONSTANT):
                plus=lambda V: V[0]*other
                result=measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'* {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        return result;
    def __rmul__(self,other):
        return measurement.__mul__(self,other)
        
    def __sub__(self,other):
        result=self+(other*-1)
        result.rename()
        return result     
    def __rsub__(self,other):
        return
        
    def __truediv__(self,other):
        if measurement.method=="Derivative": 
            from math import log
            norm_other=norm_check(self,other)
            
            mean = self.mean/norm_other.mean
            std = (self.std**2/norm_other.mean**2+
                norm_other.std**2*(self.mean*log(norm_other.mean))**2)**(1/2)
            name=self.name+'*'+norm_other.name
            result = measurement(name,mean,std)
            result.info+="Errors propgated by Derivative method"
            
         #Addition by Min-Max method
        elif measurement.method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]*V[1]
                result=measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other, measurement.CONSTANT):
                plus=lambda V: V[0]*other
                result=measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'* {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        return result;
        
    def __pow__(self,other):
        from math import log
        norm_other=norm_check(self,other)
        
        mean = self.mean**norm_other.mean
        std = ((norm_other.mean*self.mean**(norm_other.mean-1)*self.std)**2 
            + (self.mean**norm_other.mean*log(self.mean)*norm_other.std)**2)
        name=self.name+'**'+norm_other.name
        result = measurement(name,mean,std)
        #Must set correlation
        result.info+="Errors propgated by Derivative method"
        return result;

    def sin(value):
        from math import sin
        from math import cos
        mean=sin(value.mean)
        std=abs(cos(value.mean)*value.std)
        name="cos("+value.name+")"
        result=measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
        
    def cos(value):
        from math import sin
        from math import cos
        mean=cos(value.mean)
        std=abs(sin(value.mean)*value.std)
        name="sin("+value.name+")"
        result=measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
        
    def exp(value):
        from math import exp
        mean=exp(value.mean)
        std=abs(value.mean*mean*value.std)
        name="exp("+value.name+")"
        result=measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
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
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;

##############################################################################
    
    def monte_carlo(function,*args):
        #2D array
        import numpy as np
        N=len(args)
        n=measurement.mcTrials #Can be adjusted by editing measurement.mcTrials
        value=np.zeros((N,n))
        result=np.zeros(n)
        for i in range(N):
            value[i]=np.random.normal(args[i].mean,args[i].std,n)
        result=function(value)
        data=np.mean(result)
        error=np.std(result)
        argName=""
        for i in range(N):
            argName+=','+args[i].name
        name=function.__name__+"("+argName+")"
        return measurement(name,data,error)

class normalize(object):
    '''
    Creates a dummy object, similar to a measurement that can be used like a
    measurement in operations.
    '''
    def __init__(self,value):
        self.mean=value
        self.std=0
        self.name="%d"%value
   
def f(function,*args):
    N=len(args)
    mean=function(args)
    std_squared=0
    for i in range(N):
        for arg in args:        
            std_squared+=arg.std**2*partial_derivative(function,i,args)**2
    std=(std_squared)**(1/2)
    argName=""
    for i in range(N):
        argName+=','+args[i].name
    name=function.__name__+"("+argName+")"
    return measurement(name,mean,std);
      
def partial_derivative(func,var,*args):
    '''
    Expected input is normalized arguments
    '''    
    def restrict_dimension(x):
        partial_args=list(args)
        partial_args[var]=x
        return func(*partial_args);
    return derivative(restrict_dimension,args[var])

def derivative(function,point,dx=1e-10):
    return (function(point+dx)-function(point))/dx

def norm_check(arg1,arg2):
    if isinstance(arg2,arg1.__class__):
        norm_arg2=arg2
    elif isinstance(arg2, measurement.CONSTANT):
        norm_arg2=normalize(arg2)
    else:
        raise TypeError(
        "unsupported operand type(s) for -: '{}' and '{}'").format(
        arg1.__class__, type(arg2))
    return norm_arg2
