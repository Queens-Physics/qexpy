import scipy.optimize as sp
import numpy as np
import qexpy.error as qe
import qexpy.utils as qu
from math import pi

ARRAY = qu.array_types


def Rlinear(x, *pars):
    '''Linear function p[0]+p[1]*x'''
    return pars[0]+pars[1]*x

def Rpolynomial(x, *pars):
    '''Function for a polynomial of nth order, requiring n pars,
    p[0]+p[1]*x+p[2]x^2+...'''
    poly = 0
    n = 0
    for par in pars:
        poly += np.multiply(par, np.power(x, n))
        n += 1
    return poly

def Rexp(x, *pars):
    '''Function for a decaying exponential p[0]*exp(-x/p[1])'''
    return (0 if pars[1]==0 else pars[0]*np.exp(-x/pars[1]) )

def Rgauss(x, *pars):
    '''Function for a Gaussian p[2]*Gaus(p[0],p[1])'''
    #from qexpy.error import exp
    mean = pars[0]
    std = pars[1]
    norm = pars[2]
    return (0 if std==0 else norm*(2*pi*std**2)**(-0.5)*np.exp(-0.5*(x-mean)**2/std**2))

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
            self.set_fit_func(func=Rlinear,npars=2,funcname="linear",parguess=parguess)
                
        elif isinstance(model, str):
            if model in wlinear:
                self.set_fit_func(func=Rlinear,npars=2,funcname="linear",parguess=parguess)
            elif model in wgaussian:
                self.set_fit_func(func=Rgauss,npars=3,funcname="gaussian",parguess=parguess) 
            elif model in wexponential:
                self.set_fit_func(func=Rexp,npars=2,funcname="exponential",parguess=parguess) 
            elif 'pol' in model or 'Pol' in model:
                #TODO change this to regex, as it would not catch a poly of order 10 or bigger
                degree = int(model[len(model)-1]) + 1
                self.set_fit_func(func=Rpolynomial,npars=degree,funcname="polynomial",parguess=parguess)
            else:
                print("Unrecognized model string: "+model+", defaulting to linear")
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
        #TODO: catch curve_fit run time errors
           
        self.fit_pars, self.fit_pcov = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr, p0=self.parguess)

        self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))
                 
        # Use derivative method to factor x error into fit
        if xerr.nonzero()[0].size:
            yerr_eff = np.sqrt((yerr**2 + np.multiply(xerr, num_der(lambda x: self.fit_function(x, *self.fit_pars), xdata))**2))

            self.fit_pars, self.fit_pcov  = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr_eff, p0=self.parguess)
            self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))

        for i in range(self.fit_pars_err.size):
            if self.fit_pars_err[i] == float('inf'):
                self.fit_pars_err[i] = 0
            
            
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
                
        #Calculate the residuals:        
        yfit = self.fit_function(dataset.xdata, *self.fit_pars)
        yres = qe.MeasurementArray( (dataset.ydata-yfit), dataset.yerr)        
               
        return self.fit_parameters, yres
    
    
class XYDataSet:
    '''An XYDataSet contains a paired set of Measurement_Array Objects,
    typically, a set of x and y values to be used for a plot, as well
    as a method to fit that dataset. If the data set is fit multiple times
    the various fits are all recorded in a list of XYFitter objects'''
    
    #So that each dataset has a unique name (at least by default):
    unnamed_data_counter=0
    
    def __init__(self, x, y, xerr=None, yerr=None, data_name=None, xname=None, xunits=None, yname=None, yunits=None):
        '''Use MeasurementArray() to initialize a dataset'''
        if(data_name is None):
            self.name = "dataset{}".format(XYDataSet.unnamed_data_counter)
            XYDataSet.unnamed_data_counter += 1
        else:
            self.name=data_name
        
        self.x = qe.MeasurementArray(x,error=xerr, name=xname, units=xunits)
        self.xdata = self.x.get_means()
        self.xerr = self.x.get_stds()
        self.xunits = self.x.get_units_str()
        self.xname = self.x.name
        
        self.y = qe.MeasurementArray(y,error=yerr, name=yname, units=yunits)
        self.ydata = self.y.get_means()
        self.yerr = self.y.get_stds()
        self.yunits = self.y.get_units_str()
        self.yname = self.y.name
        
        if self.x.size != self.y.size:
            print("Error: x and y data should have the same number of points")
            #TODO raise an error!
        else:
            self.npoints = self.x.size
        
        self.xyfitter = []
        self.fit_pars = []
        self.fit_function = [] 
        self.fit_function_name = []
        self.fit_npars =[]
        self.yres = []
        self.nfits=0
        
    def fit(self, model=None, parguess=None, fit_range=None):
        '''Fit a data set to a model using XYFitter. Everytime this function
        is called on a data set, it adds a new XYFitter to the dataset. This
        is to allow multiple functions to be fit to the same data set'''
        fitter = XYFitter(model=model, parguess=parguess)
        fit_pars, yres = fitter.fit(self, fit_range=fit_range, fit_count=self.nfits)
        self.xyfitter.append(fitter)
        self.fit_pars.append(fit_pars) 
        self.fit_npars.append(fit_pars.size)
        self.yres.append(yres)
        self.fit_function.append(fitter.fit_function) 
        self.fit_function_name.append(fitter.fit_function_name) 
        self.nfits += 1
        return self.fit_pars[self.nfits-1]
    
    def clear_fits(self):
        self.xyfitter = []
        self.fit_pars = []
        self.fit_function = [] 
        self.fit_function_name = []
        self.fit_npars =[]
        self.yres = []
        self.nfits=0
    
    def get_x_range(self, margin=0):
        return [self.xdata.min()-self.xerr.max()-margin,\
                self.xdata.max()+self.xerr.max()+margin]
    
    def get_y_range(self, margin=0):
        return [self.ydata.min()-self.yerr.max()-margin,\
                self.ydata.max()+self.yerr.max()+margin] 
    
    def get_yres_range(self, margin=0, fitindex=-1):
        return [self.yres[fitindex].get_means().min()-self.yerr.max()-margin,\
                self.yres[fitindex].get_means().max()+self.yerr.max()+margin] 
    
def num_der(function, point, dx=1e-10):
    '''
    Returns the first order derivative of a function.
    Used in combining xerr and yerr.
    '''
    import numpy as np
    point = np.array(point)
    return np.divide(function(point+dx)-function(point), dx)   