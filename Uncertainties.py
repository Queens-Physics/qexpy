class Measurement:
    method="Derivative" #Default error propogation method
    mcTrials=10000 #option for number of trial in Monte Carlo simulation
    
    def __init__(self,name,*args):
        '''
        Creates a variable that contains a mean, standard deviation, and name
        for any value.
        '''
        import numpy
        if len(args)==2:
            self.mean=args[0]
            self.std=args[1]
            
        elif len(args)>2:
            self.mean = numpy.mean(args)
            self.std = numpy.std(args,ddof=1)
        self.name=name
        self.correlation={'Name': [name], 
            'Correlation Factor': [1]}
        self.info=""
    
    def setMethod(aMethod):
        '''
        Function to change default error propogation method used in Measurement
        functions.
        '''
        if aMethod=="Monte Carlo":
            Measurement.method="MonteCarlo"
        elif aMethod=="Min Max":
            Measurement.method="MinMax"
        else:
            Measurement.method="Derivative"
        
    def __str__(self):
        return "{:.1f}+/-{:.1f}".format(self.mean,self.std);

    def setCorrelation(self,variable,correlation):
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
    
    def __add__(self,other):
        #Addition by error propogation formula
        if Measurement.method=="Derivative":
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
        elif Measurement.method=="MinMax":
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
        
    '''
      def __sub__(self,other):
        if Measurement.method=="Derivative":           
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
            
        #Addition by Min-Max method
        elif Measurement.method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]-V[1]
                result=Measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other,int) or isinstance(other,float):
                plus=lambda V: V[0]-other
                result=Measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'+ {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        
        return result;
    '''

    def __mul__(self,other):
        if Measurement.method=="Derivative":          
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
                std = abs(mean*self.std/self.mean)
                name = self.name+"+ {:.1f}".format(float(other))
                result = Measurement(name,mean,std)
                result.setCorrelation(self,1)
         #Addition by Min-Max method
        elif Measurement.method=="MinMax":
            print("Coming Soon")
            result=self
            result.info+="Errors propogated by Min-Max method."
            
        #If method specification is bad, MC method is used
        else:
            if isinstance(other,self.__class__):
                plus=lambda V: V[0]*V[1]
                result=Measurement.MonteCarlo(plus,[self,other])
                result.rename(self.name+'+'+other.name)
            elif isinstance(other,int) or isinstance(other,float):
                plus=lambda V: V[0]*other
                result=Measurement.MonteCarlo(plus,[self])
                result.rename(self.name+'+ {:.2f}'.format(float(other)))
            result.info+="Errors propogated by Monte Carlo method"
        return result;
        
    def __sub__(self,other):
        result=self+(other*-1)
        #rename
        return result
        
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
            std = abs(mean*self.std/self.mean)
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
            std = abs(mean*other*self.std/self.mean)
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
        std=abs(value.std/value.mean)
        name="log("+value.name+")"
        result=Measurement(name,mean,std)
        result.info += "Errors propgated by Derivative method"
        #Must set correlation
        return result;
    
    def MonteCarlo(function,*args):
        #2D array
        import numpy as np
        N=len(args)
        n=Measurement.mcTrials #Can be adjusted by editing Measurement.mcTrials
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
        return Measurement(name,data,error)
