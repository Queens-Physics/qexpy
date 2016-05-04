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
        print("Please install numpy for full features.")
        numpy_installed=False
    else:
        ARRAY+=(numpy.ndarray,)
        numpy_installed=True
    
    def __init__(self,name,*args):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for inputted data.
        '''
        
        if len(args)==2 and all(isinstance(n,measurement.CONSTANT) for n in args):
            self.mean=args[0]
            self.std=args[1]
            data=None
            
        elif len(args)>2 or all(isinstance(n,measurement.ARRAY) for n in args) and len(args)==1:
            self.mean = mean(args)
            self.std = std(args,ddof=1)
            data=list(args)
        else:
            raise ValueError('''Input arguments must be one of: a mean and 
            standard deviation, an array of values, or the individual values
            themselves.''')
        self.name=name
        self.correlation={'Variable': [name], 
            'Correlation Factor': [1], 'Covariance': [self.std**2]}
        self.info={'ID': 'var%d'%(measurement.id_number), 'Formula': '', 'Method': ''
                       , 'Data': data}
        #self.ID="var%d"%(measurement.id_number)
        self.type="Uncertaintiy"
        measurement.id_number+=1
    
    def set_method(aMethod):
        '''
        Function to change default error propogation method used in measurement
        functions.
        '''
        if aMethod=="Monte Carlo":
            if measurement.numpy_installed:
                measurement.method="Monte Carlo"
            else:
                print('Numpy package must be installed to use Monte Carlo propagation, '
                      +'using the derivative method instead.')
                measurement.method="Derivative"
        elif aMethod=="Min Max":
            measurement.method="MinMax"
        else:
            measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def find_covariance(x,y):
        data_x=x.info["Data"]
        data_y=y.info["Data"]
              
        if len(data_x)!=len(data_y):
              print('Lengths of data arrays must be equal to define a covariance')
        sigma_xy=0
        for i in range(len(data_x)):
              sigma_xy+=(data_x[i]-x.mean)*(data_y[i]-y.mean)
        sigma_xy/=(len(data_x)-1)
        ro_xy=sigma_xy/x.std/y.std

        x.correlation['Variable'].append(y.name)
        x.correlation['Covariance'].append(sigma_xy)
        x.correlation['Correlation Factor'].append(ro_xy)
        y.correlation['Variable'].append(x.name)
        y.correlation['Covariance'].append(sigma_xy)
        y.correlation['Covariance Factor'].append(ro_xy)

    def set_correlation(x,y,factor):
        ro_xy=factor
        sigma_xy=ro_xy*x.std*y.std

        x.correlation['Variable'].append(y.name)
        x.correlation['Covariance'].append(sigma_xy)
        x.correlation['Correlation Factor'].append(ro_xy)
        y.correlation['Variable'].append(x.name)
        y.correlation['Covariance'].append(sigma_xy)
        y.correlation['Covariance Factor'].append(ro_xy)

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
    return norm_arg2;

def mean(*args):
    '''
    Returns the mean of an inputted array, numpy array or series of values
    '''
    args=args[0]
    print(len(args))
    return sum(args)/len(args);

def std(*args,ddof=0):
    '''
    Returns the standard deviation of an inputted array, numpy array or series of values.
    These values can have limited degrees of freedom, inputted using ddof.
    '''
    val=0
    args=args[0]
    mean=sum(args)/len(args)
    for i in range(len(args)):
        val+=(args[i]-mean)**2
    val/=(len(args)-ddof)
    return val;
    
