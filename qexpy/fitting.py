import scipy.optimize as sp
import numpy as np
import qexpy.error as qe
import qexpy.utils as qu
from math import pi

ARRAY = qu.array_types


def Rlinear(x, *pars):
    return pars[0]+pars[1]*x

def Rpolynial(x, *pars):
    '''Function for a polynomial of nth order, requiring n pars.'''
    poly = 0
    n = 0
    for par in pars:
        poly += np.multiply(par, np.power(x, n))
        n += 1
    return poly

def Rexp(x, *pars):
    '''Function for a Gaussian'''
    from error import exp
    return (0 if pars[1]==0 else pars[0]*exp(-x/pars[1]) )

def Rgauss(x, *pars):
    '''Function for a Gaussian'''
    from error import exp
    mean = pars[0]
    std = pars[1]
    norm = pars[2]
    return (0 if std==0 else norm*(2*pi*std**2)**(-0.5)*exp(-0.5*(x-mean)**2/std**2))

class XYFitter:
    '''A class to fit a set of data given as y(x)'''
    
    def __init__(self, model = None, parguess=None):
        self.xydataset=None   
        self.initialize_fit_function(model, parguess)
        
    def set_fit_func(self, func, npars, funcname=None, parguess=None):
        '''Set the fit function and the number of parameter, given
        the expected number of parameters, npars'''
        
        self.fit_function=func
        self.fit_npars=npars
        self.fit_function_name="custom" if funcname is None else funcname
        if parguess is None or len(parguess)!=npars:
            self.parguess = npars*[1]
        else:
            self.parguess = parguess
                
        #self.fit_pars = MeasurementArray(self.fit_npars)
           
    def initialize_fit_function(self, model=None, parguess=None):
        '''Set the model and parameter guess'''
        
        wlinear = ('linear', 'Linear', 'line', 'Line',)
        wgaussian = ('gaussian', 'Gaussian', 'Gauss', 'gauss', 'normal',)
        wexponential = ('exponential', 'Exponential', 'exp', 'Exp',)
        
        if model is None:
            self.set_fit_func(func=RLinear,npars=2,funcname="linear",parguess=parguess)
                
        elif isinstance(model, str):
            if model in wlinear:
                self.set_fit_func(func=Rlinear,npars=2,funcname="linear",parguess=parguess)
            elif model in wgaussian:
                self.set_fit_func(func=Rgauss,npars=3,funcname="gaussian",parguess=parguess) 
            elif model in wexponential:
                self.set_fit_func(func=Rexp,npars=2,funcname="exponential",parguess=parguess) 
            elif model[:3] is 'pol' or model[:3] is 'Pol':
                #TODO change this to regex, as it would not catch a poly of order 10 or bigger
                degree = int(model[len(model)-1]) + 1
                self.set_fit_func(func=Rexp,npars=degree,funcname="polynomial",parguess=parguess)
            else:
                print("Unrecognized model string, defaulting to linear")
                self.set_fit_func(func=Rlinear,npars=2,funcname="linear",parguess=parguess)
        else:
            import inspect
            if not inspect.isfunction(model):
                print("Error: model function should be in form: def model(x, *pars)")
                return
            argspec = inspect.getargspec(model)
            if len(argspec[0])!=1:
                print("Error: model function should be in form: def model(x, *pars)")
                return
            if argspec[1] is None:
                print("Error: model function should be in form: def model(x, *pars)")
                return
            if parguess is None:
                print("Error: must specify a guess for a custom function")
                return
            
            self.set_fit_func(func=model, npars=len(parguess), funcname="custom", parguess=parguess)

    def fit(self, dataset, fit_range=None, fit_count=0):
        ''' Perform a fit of the fit_function to a data set'''
        if self.fit_function is None:
            print("Error: fit function not set!")
            return
        
        #Grab the data
        xdata = dataset.xdata
        ydata = dataset.ydata
        xerr = dataset.xerr
        yerr = dataset.yerr
        
        #If user specified a fit range, reduce the data:    
        if type(fit_range) in ARRAY and len(fit_range) is 2:
            indices = np.where(np.logical_and(xdata>=fit_range[0], xdata<=fit_range[1]))
            xdata=xdata[indices]
            ydata=ydata[indices]
            xerr=xerr[indices]
            yerr=yerr[indices]
            
        #if the x errors are not zero, convert them to equivalent errors in y
        #TODO: check the math on this...
            
        self.fit_pars, self.fit_pcov = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr, p0=self.parguess)

        self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))
         
        # Use derivative method to factor x error into fit
        if xerr.nonzero()[0].size:
            yerr_eff = np.sqrt((yerr**2 + np.multiply(xerr, num_der(lambda x: model(x, *self.fit_pars), xdata))**2))

            self.fit_pars, self.fit_pcov  = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr_eff, p0=self.parguess)
            self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))

        self.fit_npars = self.fit_pars.size  
        
        parnames = dataset.name+"_"+self.fit_function_name+"_fit{}".format(fit_count)+"_fitpars"
        self.fit_parameters = qe.MeasurementArray(self.fit_npars,name=parnames)

        for i in range(self.fit_npars):
            if self.fit_function_name is 'gaussian':
                if i is 0:
                    name = 'mean'
                elif i is 1:
                    name = 'standard deviation'
                elif i ==2:
                    name = 'normalization'               
            elif self.fit_function_name is 'linear':
                if i is 0:
                    name = 'intercept'
                elif i is 1:
                    name = 'slope'
            else:
                name = 'par_%d' % (i)
            name = parnames +"_"+name
            self.fit_parameters[i]= qe.Measurement(self.fit_pars[i], self.fit_pars_err[i], name=name)
                 
        for i in range(self.fit_npars):
            for j in range(i+1, self.fit_npars):
                self.fit_parameters[i].set_covariance(self.fit_parameters[j],
                                                      self.fit_pcov[i][j])       

        return self.fit_parameters
    
    
class XYDataSet:
    '''An XYDataSet contains a paired set of Measurement_Array Objects,
    typically, a set of x and y values to be used for a plot'''
    unnamed_data_counter=0
    def __init__(self, x, y, xerr=None, yerr=None, data_name=None):
        if(data_name is None):
            self.name = "dataset{}".format(XYDataSet.unnamed_data_counter)
            XYDataSet.unnamed_data_counter += 1
        else:
            self.name=data_name
        
        self.x = qe.MeasurementArray(x,error=xerr)
        self.xdata = self.x.get_means()
        self.xerr = self.x.get_stds()
        self.xunits = self.x.get_units_str()
        self.xname = self.x.name
        
        self.y = qe.MeasurementArray(y,error=yerr)
        self.ydata = self.y.get_means()
        self.yerr = self.y.get_stds()
        self.yunits = self.y.get_units_str()
        self.yname = self.y.name
        
        self.xyfitter = []
        self.fit_pars = []
        self.fit_function = []       
        self.nfits=0
        
    def fit(self, model=None, parguess=None, fit_range=None):
        '''Fit a data set to a model using XYFitter. Everytime this function
        is called on a data set, it adds a new XYFitter to the dataset. This
        is to allow multiple functions to be fit to the same data set'''
        fitter = XYFitter(model=model, parguess=parguess)
        fit_pars = fitter.fit(self, fit_range=fit_range, fit_count=self.nfits)
        self.xyfitter.append(fitter)
        self.fit_pars.append(fit_pars) # redundant, just for convenience
        self.fit_function.append(fitter.fit_function) #redundant, just for convenience
        self.nfits += 1
        return self.fit_pars[self.nfits-1]
   