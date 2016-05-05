class measurement:
    method="Derivative" #Default error propogation method
    mcTrials=10000 #option for number of trial in Monte Carlo simulation
    id_number=1
    
    #Defining common types under single array
    #Testing GitHub
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
    
    def __init__(self,*args,name=None):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for inputted data.
        '''
        
        if len(args)==2 and all(isinstance(n,measurement.CONSTANT)\
                for n in args):
            self.mean=args[0]
            self.std=args[1]
            data=None
            
        elif all(isinstance(n,measurement.ARRAY) for n in args) and \
                len(args)==1:
            args=args[0]
            self.mean = mean(args)
            self.std = std(args,ddof=1)
            data=list(args)
        elif len(args)>2:
            self.mean = mean(args)
            self.std = std(args,ddof=1)
            data=list(args)
        else:
            raise ValueError('''Input arguments must be one of: a mean and 
            standard deviation, an array of values, or the individual values
            themselves.''')
        if name is not None:
            self.name=name
        else:
            self.name='var%d'%(measurement.id_number)
        self.correlation={'Variable': [name], 
            'Correlation Factor': [1], 'Covariance': [self.std]}
        self.info={'ID': 'var%d'%(measurement.id_number), 'Formula': \
        'var%d'%(measurement.id_number) ,'Method': '', 'Data': data}
        self.first_der={self.info['ID']:1}
        self.type="Uncertaintiy"
        measurement.id_number+=1
    
    def set_method(aMethod):
        '''
        Function to change default error propogation method used in 
        measurement functions.
        '''
        mc_list=('MC','mc','montecarlo','Monte Carlo','MonteCarlo',\
                'monte carlo',)
        min_max=('Min Max','MinMax','minmax','min max')
        
        if aMethod in mc_list:
            if measurement.numpy_installed:
                measurement.method="Monte Carlo"
            else:
                print('Numpy package must be installed to use Monte Carlo \
                        propagation, using the derivative method instead.')
                measurement.method="Derivative"
        elif aMethod in min_max:
            measurement.method="MinMax"
        else:
            measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def find_covariance(x,y):
        data_x=x.info["Data"]
        data_y=y.info["Data"]
              
        if len(data_x)!=len(data_y):
              print('Lengths of data arrays must be equal to\
                      define a covariance')
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
        y.correlation['Correlation Factor'].append(ro_xy)

    def set_correlation(x,y,factor):
        ro_xy=factor
        sigma_xy=ro_xy*x.std*y.std

        x.correlation['Variable'].append(y.name)
        x.correlation['Covariance'].append(sigma_xy)
        x.correlation['Correlation Factor'].append(ro_xy)
        y.correlation['Variable'].append(x.name)
        y.correlation['Covariance'].append(sigma_xy)
        y.correlation['Covariance Factor'].append(ro_xy)

    def get_correlation(self,variable):
        #Duplicating Correlation
        if self.info['Data'] is not None \
                and variable.info['Data'] is not None \
                and all(self.correlation['Variable'][i]!=variable.name \
                for i in range(len(self.correlation['Variable']))):
            measurement.find_covariance(self,variable)
        if any(self.correlation['Variable'][i]==variable.name
                   for i in range(len(self.correlation['Variable']))):
            for j in range(len(self.correlation["Variable"])):
                if self.correlation["Variable"][j]==variable.name:
                    index=j
                    return self.correlation["Covariance"][index]
        else:
            return 0;
        
    def rename(self,newName):
        self.name=newName
    
    def update_info(self, operation, var1, var2=None, func_flag=None):
        '''
        Function to update the name, formula and method of a value created
        by a measurement operation. The name is updated by combining the names
        of the object acted on with another name using whatever operation is
        being performed. The function flag is to change syntax for functions
        like sine and cosine. Method is updated by acessing the class 
        property.
        '''
        if func_flag is None and var2 is not None:
            self.rename(var1.name+operation+var2.name)
            self.info['Formula']=var1.info['Formula']+operation+\
                    var2.info['Formula']
        elif func_flag is not None:
            self.rename(operation+'('+var1.name+')')
            self.info['Formula']=operation+'('+var1.info['Formula']+')'
            self.info['Method']+="Errors propagated by "+measurement.method+\
                    ' method.\n'
        else:
            print('Something went wrong in update_info')
            
    def d(self,variable=None):
        '''
        Function to find the derivative of a measurement or measurement like
        object. By default, derivative is with respect to itself, which will
        always yeild 1. Operator acts by acessing the self.first_der 
        dictionary and returning the value stored there under the specific
        variable inputted (ie. deriving with respect to variable=???)
        '''
        if not hasattr(variable,'type'):
            return 'Only measurement objects can be derived.'
        
        if variable is None:
            variable=self.info['ID']
        elif variable.info['ID'] not in self.first_der:
            self.first_der[variable.info['ID']]=0
        derivative=self.first_der[variable.info["ID"]]
        return derivative
    
    def check_der(self,b):
        '''
        Checks the existance of the derivative of an object in the 
        dictionary of said object with respect to another variable, given
        the ID key of the other variable.
        '''
        for key in b.first_der:
            if key in self.first_der:
                pass;
            else:
                self.first_der[key]=0
        
#######################################################################
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
        result=sub(other,self)
        result.rename(other.name+'-'+self.name)
        
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

#######################################################################
    
    def monte_carlo(function,*args):
        #2D array
        import numpy as np
        N=len(args)
        n=measurement.mcTrials #Can be adjusted in measurement.mcTrials
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
    value=measurement(value,0,name='%d'%value)
    value.info['ID']='Constant'
    value.first_der={value.info['ID']:0}
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
    return sum(args)/len(args);

def std(*args,ddof=0):
    '''
    Returns the standard deviation of an inputted array, numpy array or 
    series of values. These values can have limited degrees of freedom, 
    inputted using ddof.
    '''
    val=0
    args=args[0]
    mean=sum(args)/len(args)
    for i in range(len(args)):
        val+=(args[i]-mean)**2
    std=(val/(len(args)-1))**(1/2)
    return std;
    
