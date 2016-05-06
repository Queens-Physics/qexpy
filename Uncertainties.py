class measurement:
    method="Derivative" #Default error propogation method
    mcTrials=10000 #option for number of trial in Monte Carlo simulation
    id_number=1 #Instances var0
    
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
        self.covariance={'Name': [self.name], 'Covariance': [self.std**2]}
        self.info={'ID': 'var%d'%(measurement.id_number), 'Formula': \
        'var%d'%(measurement.id_number) ,'Method': '', 'Data': data}
        self.first_der={self.info['ID']:1}
        self.type="measurement"
        measurement.id_number+=1
    
    def set_method(chosen_method):
        '''
        Choose the method of error propagation to be used. Enter a string.        
        
        Function to change default error propogation method used in 
        measurement functions.
        '''
        mc_list=('MC','mc','montecarlo','Monte Carlo','MonteCarlo',\
                'monte carlo',)
        min_max_list=('Min Max','MinMax','minmax','min max',)
        derr_list=('Derivative', 'derivative','diff','der',)
        default = 'Derivative'        
        
        if chosen_method in mc_list:
            if measurement.numpy_installed:
                measurement.method="Monte Carlo"
            else:
                print('Numpy package must be installed to use Monte Carlo \
                        propagation, using the derivative method instead.')
                measurement.method="Derivative"
        elif chosen_method in min_max_list:
            measurement.method="Min Max"
        elif chosen_method in derr_list:
            measurement.method="Derivative"
        else:
            print("Method not recognized, using"+default+"method.")
            measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def find_covariance(x,y):
        '''
        Uses the data from which x and y were generated to calculate
        covariance and add this informaiton to x and y.
        
        Requires data arrays to be stored in the .info of both objects
        and that these arrays are of the same length, as the covariance is
        only defined in these cases.
        '''
        data_x=x.info["Data"]
        data_y=y.info["Data"]
              
        if len(data_x)!=len(data_y):
              print('Lengths of data arrays must be equal to\
                      define a covariance')
        sigma_xy=0
        for i in range(len(data_x)):
              sigma_xy+=(data_x[i]-x.mean)*(data_y[i]-y.mean)
        sigma_xy/=(len(data_x)-1)

        x.covariance['Name'].append(y.name)
        x.covariance['Covariance'].append(sigma_xy)
        y.covariance['Name'].append(x.name)
        y.covariance['Covariance'].append(sigma_xy)

    def set_correlation(x,y,factor):
        '''
        Manually set the correlation between two quantities
        
        Given a correlation factor, the covariance and correlation
        between two variables is added to both objects.
        '''
        ro_xy=factor
        sigma_xy=ro_xy*x.std*y.std

        x.covariance['Name'].append(y.name)
        x.covariance['Covariance'].append(sigma_xy)
        y.covariance['Name'].append(x.name)
        y.covariance['Covariance'].append(sigma_xy)

    def get_covariance(self,variable):
        '''
        Returns the covariance of the object and a specified variable.
        
        This funciton checks for the existance of a data array in each 
        object and that the covariance of the two objects is not already
        specified. In each case, the covariance is returned, unless
        the data arrays are of different lengths or do not exist, in that
        case a covariance of zero is returned.        
        '''
        if self.info['Data'] is not None \
                and variable.info['Data'] is not None \
                and all(self.covariance['Name'][i]!=variable.name \
                for i in range(len(self.covariance['Name']))):
            measurement.find_covariance(self,variable)
        if any(self.covariance['Name'][i]==variable.name
                   for i in range(len(self.covariance['Name']))):
            for j in range(len(self.covariance["Name"])):
                if self.covariance["Name"][j]==variable.name:
                    index=j
                    return self.covariance["Covariance"][index]
        else:
            return 0;
    
    def get_correlation(x,y):
        '''
        Returns the correlation factor of two measurements.
        
        Using the covariance, or finding the covariance if not defined,
        the correlation factor of two measurements is returned.        
        '''
        if y.name in x.covariance['Name']:
            pass
        else:
            measurement.find_covariance(x,y)
        sigma_xy=x.covariance[y.name]
        sigma_x=x.covariance[x.name]
        sigma_y=y.covariance[y.name]
        return sigma_xy/sigma_x/sigma_y    
        
    def rename(self,newName):
        '''
        Renames an object, requires a string.
        '''
        self.name=newName
    
    def _update_info(self, operation, var1, var2=None, func_flag=None):
        '''
        Update the formula, name and method of an object.        
        
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
        Returns the numerical value of the derivative with respect to an 
        inputed variable.        
        
        Function to find the derivative of a measurement or measurement like
        object. By default, derivative is with respect to itself, which will
        always yeild 1. Operator acts by acessing the self.first_der 
        dictionary and returning the value stored there under the specific
        variable inputted (ie. deriving with respect to variable=???)
        '''
        if variable is not None \
                and not hasattr(variable,'type'):
            return 'Only measurement objects can be derived.'
        elif variable is None:
            return self.first_der
        if variable.info['ID'] not in self.first_der:
            self.first_der[variable.info['ID']]=0
        derivative=self.first_der[variable.info["ID"]]
        return derivative
    
    def check_der(self,b):
        '''
        Checks for a derivative with respect to b, else zero is defined as
        the derivative.        
        
        Checks the existance of the derivative of an object in the 
        dictionary of said object with respect to another variable, given
        the variable itself, then checking for the ID of said variable
        in the .first_der dictionary. If non exists, the deriviative is 
        assumed to be zero.
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
        result.rename("Need to fix naming")
        return result
        
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
        
    def __neg__(self):
        return measurement(-self.mean,self.std,name='-'+self.name)

#######################################################################
    
    def monte_carlo(function,*args):
        '''
        Uses a Monte Carlo simulation to determine the mean and standard 
        deviation of a function.
        
        Inputted arguments must be measurement type. Constants can be used
        as 'normalized' quantities which produce a constant row in the 
        matrix of normally randomized values.
        '''
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
    '''
    Returns a measurement object that is constant with no deviation.
    
    Object can be acted on like a measurement but does not increase the 
    ID counter and is labelled as a constant. The derivative of this
    object with respect to anything is zero.
    '''
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
    Returns the parital derivative of a dunction with respect to var.
    
    This function wraps the inputted function to become a function
    of only one variable, the derivative is taken with respect to said
    variable.
    '''    
    def restrict_dimension(x):
        partial_args=list(args)
        partial_args[var]=x
        return func(*partial_args);
    return derivative(restrict_dimension,args[var])

def derivative(function,point,dx=1e-10):
    '''
    Returns the first order derivative of a function.
    '''
    return (function(point+dx)-function(point))/dx

def mean(*args):
    '''
    Returns the mean of an inputted array, numpy array or series of values
    '''
    args=args[0]
    return sum(args)/len(args);

def std(*args,ddof=0):
    '''
    Returns the standard deviation of an inputted array.
    
    Array types include a list, numpy array or 
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
    
