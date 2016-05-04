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
        print("Please install numpy for full features."
    else:
        ARRAY+=(numpy.ndarray,)
    
    def __init__(self,name,*args):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for inputted data.
        '''
        
        if len(args)==2 and all(isinstance(n,measurement.CONSTANT) for n in args):
            self.mean=args[0]
            self.std=args[1]
            
        elif len(args)>2 or all(isinstance(n,measurement.ARRAY) for n in args
                                            ) and len(args)==1:
            self.mean = numpy.mean(args)
            self.std = numpy.std(args,ddof=1)
            data=list(args)
        else:
            raise ValueError('''Input arguments must be one of: a mean and 
            standard deviation, an array of values, or the individual values
            themselves.''')
        self.name=name
        self.correlation={'Name': [name], 
            'Correlation Factor': [1]}
        self.info={'ID': 'var%d'%(measurement.id_number), 'Formula': '', 'Method': ''
                       , 'Raw Data': data}
        #self.ID="var%d"%(measurement.id_number)
        self.type="Uncertaintiy"
        measurement.id_number+=1
    
    def set_method(aMethod):
        '''
        Function to change default error propogation method used in measurement
        functions.
        '''
        if aMethod=="Monte Carlo":
            measurement.method="Monte Carlo"
        elif aMethod=="Min Max":
            measurement.method="MinMax"
        else:
            measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def set_covariance(data1,data2):
        
    
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
        from operations import add
        return add(self,other)
    def __radd__(self,other):
        from operations import add
        return add(self,other);  

    def __mul__(self,other):
        from operations import mul
        return mul(self,other);
    def __rmul__(self,other):
        from operations import mul
        return mul(self,other);
        
    def __sub__(self,other):
        from operations import sub
        return sub(self,other);
    def __rsub__(self,other):
        from operations import sub
        return sub(other,self);
        
    def __truediv__(self,other):
        from operations import div
        return div(self,other);
    def __rtruediv__(self,other):
        from operations import div
        return div(other,self);
        
    def __pow__(self,other):
        from operations import power
        return power(self,other)
    def __rpow__(self,other):
        from operations import power
        return power(other,self)

##############################################################################
    
    def monte_carlo(function,*args):
        #2D array
        import numpy as np
        N=len(args)
        n=measurement.mcTrials #Can be adjusted by editing measurement.mcTrials
        value=np.zeros((N,n))
        result=np.zeros(n)
        for i in range(N):
            if args[i].std==0:
                value[i]=args[i].mean
            else:
                value[i]=np.random.normal(args[i].mean,args[i].std,n)
        result=function(*value)
        data=np.mean(result)
        error=np.std(result)
        argName=""
        for i in range(N):
            argName+=','+args[i].name
        name=function.__name__+"("+argName+")"
        return measurement(name,data,error)

def normalize(value):
    value=measurement('%d'%value,value,0)
    value.ID='N/A'
    measurement.id_number-=1
    return value;
   
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
        "unsupported operand type(s) for -: '{}' and '{}'".format(
        type(arg1), type(arg2)))
    return norm_arg2
