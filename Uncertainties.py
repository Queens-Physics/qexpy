class Measurement:
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
        if isinstance(other,self.__class__):
            mean = self.mean+other.mean
            std = self.std+other.std
            name=self.name+'+'+other.name
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.setCorrelation(other,1)
            result.info+="Errors propgated by derrivitive method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean+other
            std = self.std
            name=self.name+'+ constant'
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.info+="Errors propgated by derrivitive method"
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
            ).format(self.__class__, type(other))
        return result;
        
    def __sub__(self,other):
        if isinstance(other,self.__class__):
            mean = self.mean-other.mean
            std = self.std+other.std
            name=self.name+'-'+other.name
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.setCorrelation(other,-1)
            result.info+="Errors propgated by derrivitive method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean-other
            std = self.std
            name=self.name+'- constant'
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
            result.info+="Errors propgated by derrivitive method"
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
            result.info+="Errors propgated by derrivitive method"
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
            result.info += "Errors propgated by derrivitive method"
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
        import math
        if isinstance(other,self.__class__):
            mean = self.mean**other.mean
            std = ((other.mean*self.mean**(other.mean-1)*self.std)**2 
                + (self.mean**other.mean*math.log(self.mean)*other.std)**2)
            name=self.name+'**'+other.name
            result = Measurement(name,mean,std)
            #Must set correlation
            result.info+="Errors propgated by derrivitive method"
        elif isinstance(other,int) or isinstance(other,float):
            mean = self.mean**other
            std = mean*other*self.std/self.mean
            name = self.name+"+ {:.1f}".format(float(other))
            result = Measurement(name,mean,std)
            result.setCorrelation(self,1)
        return result;

    def sin(mValue):
        from math import sin
        from math import cos
        mean=sin(mValue.mean)
        std=abs(cos(mValue.mean)*mValue.std)
        name="cos("+mValue.name+")"
        result=Measurement(name,mean,std)
        #Must set correlation



#Test Code
x=Measurement('x',10,1)
y=Measurement('y',[10,15,20])
