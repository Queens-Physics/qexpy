class Measurement:
    method="Derivative"
    def __init__(self,name,data,error='null',correlation=1):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for any value.
        '''
        import numpy
        if error!='null':
            self.mean=data
            self.std=error
            
        elif type(data)==list:
            self.mean = numpy.mean(data)
            self.std = numpy.std(data,ddof=1)
        self.name=name
        self.correlation={'Name': [name], 
            'Correlation Factor': [correlation]}
        self.info=""
    
    def setMethod(method):
        if method=="Derivative":
            global Method
            Method=method
        elif method=="MinMax":
            global Method
            Method=method
        else:
            global Method
            Method="MonteCarlo"
    setMethod("Derivative")
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def setCorrelation(self,variable,correlation):
        '''
        Adds a correlation factor between the variable being acted on and 
        any specified variable. Correlation factor must be specified.
        '''
        self.correlation['Name'].append(variable.name)
        self.correlation['Correlation Factor'].append(correlation)
        
        y.correlation['Name'].append(self.name)
        y.correlation['Correlation Factor'].append(correlation)
    
    def rename(self,newName):
        self.name=newName
    
    def __add__(self,other):
        #Addition by error propogation formula
        if Method=="Derivative":
            if isinstance(other,self.__class__):
                mean = self.mean+other.mean
                std = self.std+other.std
                name=self.name+'+'+other.name
                result = Measurement(name,mean,std)
                result.setCorrelation(self,1)
                result.setCorrelation(other,1)
            elif isinstance(other,int) or isinstance(other,float):
                mean = self.mean+other
                std = self.std
                name=self.name+'+ {:.2f}'.format(float(other))
                result = Measurement(name,mean,std)
                result.setCorrelation(self,1)
            else:
                raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
            result.info+="Errors propgated by Derivative method"
            
        #Addition by Min-Max method
        elif Method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]+V[1]
                result=Measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other,int) or isinstance(other,float):
                plus=lambda V: V[0]+other
                result=Measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'+ {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        return result;
        
    def __sub__(self,other):
        if isinstance(other,self.__class__):
            mean = self.mean-other.mean
            std = self.std+other.std
            name=self.name+'-'+other.name
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.setCorrelation(other,-1)
            result.info+="Errors propgated by Derivative method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean-other
            std = self.std
            name=self.name+'- constant'
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.info+="Errors propgated by Derivative method"
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
            ).format(self.__class__, type(other))
        return result;

    def __mul__(self,other):
        if isinstance(other,self.__class__):
            mean = self.mean*other.mean
            std = mean*(self.std/self.mean + other.std/other.mean)
            name=self.name+'*'+other.name
            result = Measurement(name,mean,std)
            result.setCorrelation(self,0.5)
            result.setCorrelation(other,0.5)
            result.info+="Errors propgated by Derivative method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean*other
            std = mean*self.std/self.mean
            name = self.name+"+ {:.1f}".format(float(other))
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
        return result;
        
    def __truediv__(self,other):
        if isinstance(other,self.__class__):
            mean = self.mean/other.mean
            std = mean*(self.std/self.mean + other.std/other.mean)
            name = self.name+'/'+other.name
            result = Measurement(name,mean,std)
            #Must set correlation
            result.info += "Errors propgated by Derivative method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean/other
            std = mean*self.std/self.mean
            name = self.name+"+ {:.1f}".format(float(other))
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
            ).format(self.__class__, type(other))
        return result;
        
    def __pow__(self,other):
        from math import log
        if isinstance(other,self.__class__):
            mean = self.mean**other.mean
            std = ((other.mean*self.mean**(other.mean-1)*self.std)**2 
                + (self.mean**other.mean*log(self.mean)*other.std)**2)
            name=self.name+'**'+other.name
            result = Measurement(name,mean,std)
            #Must set correlation
            result.info+="Errors propgated by Derivative method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean**other
            std = mean*other*self.std/self.mean
            name = self.name+"+ {:.1f}".format(float(other))
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
        return result;

    def sin(value):
        from math import sin
        from math import cos
        mean=sin(value.mean)
        std=abs(cos(value.mean)*value.std)
        name="cos("+value.name+")"
        result=Measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
        
    def cos(value):
        from math import sin
        from math import cos
        mean=cos(value.mean)
        std=abs(sin(value.mean)*value.std)
        name="sin("+value.name+")"
        result=Measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
        
    def exp(value):
        from math import exp
        mean=exp(value.mean)
        std=abs(value.mean*mean*value.std)
        name="exp("+value.name+")"
        result=Measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
    
    def e(value):
        Measurement.exp(value)
        
    def log(value):
        from math import log
        mean=log(value.mean)
        std=value.std/value.mean
        name="log("+value.name+")"
        result=Measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
    
    def MonteCarlo(function,*argument):
        #2D array
        import numpy as np
        N=np.size(argument)
        n=10000
        value=np.zeros(N)
        result=np.zeros(n)
        for i in range(n):
            for j in range(N):
                value[j]=np.random.normal(argument[j].mean,argument[j].std)
            result[i]=function(value)
        data=np.mean(result)
        error=np.std(result)
        argName=""
        for i in range(N):
            argName+=argument[i].name
        name=function.__name__+"("+argName+")"
        return Measurement(name,data,error)


#Test Code
x=Measurement('x',10,1)
y=Measurement('y',[10,15,20])
def f(x):
    return x[0]*x[1];