import scipy.optimize as sp
import numpy as np
import qexpy as q
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
    
    for i in range(len(pars)):
        poly += pars[i]*x**i
        
    return poly

def Rexp(x, *pars):
    '''Function for a decaying exponential p[0]*exp(-x/p[1])'''
    return (0 if pars[1]==0 else pars[0]/pars[1]*np.exp(-x/pars[1]) )

def Rgauss(x, *pars):
    '''Function for a Gaussian p[2]*Gaus(p[0],p[1])'''
    #from qexpy.error import exp
    mean = pars[0]
    std = pars[1]
    norm = pars[2]
    return (0 if std==0 else norm*(2*pi*std**2)**(-0.5)*np.exp(-0.5*(x-mean)**2/std**2))

class XYFitter:
    '''A class to fit an XYDatatset to a function/model using scipy.optimize'''
    
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
        
        nz = np.count_nonzero(yerr)
        if nz < ydata.size and nz != 0:
            print("Warning: some errors on data are zero, switching to MC errors")
            yerr = dataset.y.get_stds(method="MC")
            #now, check again
            nz = np.count_nonzero(yerr)    
            if nz < ydata.size and nz != 0:
                print("Error: Some MC errors are zero as well, I give up")
                return None
            #We're ok, modify the errors in the dataset to be the MC ones
            dataset.yerr = yerr
        
        #If user specified a fit range, reduce the data:    
        if type(fit_range) in ARRAY and len(fit_range) is 2:
            indices = np.where(np.logical_and(xdata>=fit_range[0], xdata<=fit_range[1]))
            xdata=xdata[indices]
            ydata=ydata[indices]
            xerr=xerr[indices]
            yerr=yerr[indices]
            
        #if the x errors are not zero, convert them to equivalent errors in y
        #TODO: check the math on this...
        
        #The maximum number of function evaluations
        maxfev = 200 *(xdata.size+1) if q.settings["fit_max_fcn_calls"] == -1 else q.settings["fit_max_fcn_calls"]
        try:
            self.fit_pars, self.fit_pcov = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr, p0=self.parguess,
                                                    maxfev=maxfev)

            self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))
        except RuntimeError:
            print("Error: Fit could not converge; are the y errors too small? Is the function defined?")
            print("Is the parameter guess good?")
            return None
                 
        # Use derivative method to factor x error into fit
        if xerr.nonzero()[0].size:
            yerr_eff = np.sqrt((yerr**2 + np.multiply(xerr, num_der(lambda x: self.fit_function(x, *self.fit_pars), xdata))**2))

            try:
                self.fit_pars, self.fit_pcov  = sp.curve_fit(self.fit_function, xdata, ydata,
                                                    sigma=yerr_eff, p0=self.parguess,
                                                    maxfev=maxfev)
                self.fit_pars_err = np.sqrt(np.diag(self.fit_pcov))
            except RuntimeError:
                print("Error: Fit could not converge; are the y errors too small? Is the function defined?")
                print("Is the parameter guess good?")
                return None
            
        #this should already be true, but let's be sure:
        self.fit_npars=self.fit_pars.size
        
        #This is to catch the case of scipy.optimize failing to determin
        #the covariance matrix

        for i in range(self.fit_npars):
            if self.fit_pars_err[i] == float('inf') or self.fit_pars_err[i] == float('nan'):
                #print("Warning: Error for fit parameter",i,"cannot be trusted")
                self.fit_pars_err[i] = 0
            for j in range(self.fit_npars):
                if self.fit_pcov[i][j] == float('inf') or self.fit_pcov[i][j] == float('nan'):
                    #print("Warning: Covariance between parameters",i,j,"cannot be trusted")
                    self.fit_pcov[i][j]=0.
 
        
        parnames = dataset.name+"_"+self.fit_function_name+"_fit{}".format(fit_count)+"_fitpars"
        self.fit_parameters = qe.MeasurementArray(self.fit_npars,name=parnames)

        for i in range(self.fit_npars):
            if self.fit_function_name is 'gaussian':
                if i is 0:
                    name = 'mean'
                elif i is 1:
                    name = 'sigma'
                elif i ==2:
                    name = 'normalization'               
            elif self.fit_function_name is 'linear':
                if i is 0:
                    name = 'intercept'
                elif i is 1:
                    name = 'slope'
            elif self.fit_function_name is 'exponential':
                if i is 0:
                    name = 'amplitude'
                elif i is 1:
                    name = 'decay-constant'
            else:
                name = 'par%d' % (i)
            name = parnames +"_"+name
            self.fit_parameters[i]= qe.Measurement(self.fit_pars[i], self.fit_pars_err[i], name=name)
                 
        for i in range(self.fit_npars):
            for j in range(i+1, self.fit_npars):
                self.fit_parameters[i].set_covariance(self.fit_parameters[j],
                                                      self.fit_pcov[i][j])   
                
        #Calculate the residuals:        
        yfit = self.fit_function(dataset.xdata, *self.fit_pars)
        self.fit_yres = qe.MeasurementArray( (dataset.ydata-yfit), dataset.yerr)  
        #Calculate the chi-squared:
        self.fit_chi2 = 0
        for i in range(xdata.size):
            if self.fit_yres[i].std !=0:
                self.fit_chi2 += (self.fit_yres[i].mean/self.fit_yres[i].std)**2
        self.fit_ndof = self.fit_yres.size-self.fit_npars-1
               
        return self.fit_parameters
        
        

def DataSetFromFile(filename, xcol=0, ycol=1, xerrcol=2, yerrcol=3, delim= ' ',
                    data_name=None, xname=None, xunits=None, yname=None, yunits=None,
                    is_histogram=False):
    '''Create a DatatSet from a file, where the data is organized into 4 columns delimited
    by delim. User can specify which columns contain what information, the default is
    x,y,xerr,yerr. User MUST specify if xerr or yerr are missing by setting those columns to
    'None', and the method will automatically assign error of zero.
     '''
    data = np.loadtxt(filename, delimiter=delim)
    xdata = data[:,xcol]
    
    if not is_histogram:
        _ydata = data[:,ycol]
        if xerrcol!= None:
            xerrdata = data[:,xerrcol]
        else:
            xerrdata = np.zeros(xdata.size)
        
        if yerrcol!= None:
            yerrdata = data[:,yerrcol]
        else:
            yerrdata = np.zeros(ydata.size)
    else:
        _ydata=None
         
    return XYDataSet(xdata, ydata=_ydata, xerr=xerrdata, yerr=yerrdata, data_name=data_name,
                     xname=xname, xunits=xunits, yname=yname, yunits=yunits,
                     is_histogram=is_histogram)
    
class XYDataSet:
    '''An XYDataSet contains a paired set of Measurement_Array Objects,
    typically, a set of x and y values to be used for a plot, as well
    as a method to fit that dataset. If the data set is fit multiple times
    the various fits are all recorded in a list of XYFitter objects.
    One can also construct an XYDatatSet from histogram data, which then
    gets converted to equivalent X and Y measurements.
    '''
    
    #So that each dataset has a unique name (at least by default):
    unnamed_data_counter=0
    
    def __init__(self, xdata, ydata=None, xerr=None, yerr=None, data_name=None,
                 xname=None, xunits=None, yname=None, yunits=None,
                 is_histogram=False, bins=50):
        '''Use MeasurementArray() to initialize a dataset'''
        if(data_name is None):
            self.name = "dataset{}".format(XYDataSet.unnamed_data_counter)
            XYDataSet.unnamed_data_counter += 1
        else:
            self.name=data_name
            
        if ydata is None and not is_histogram:
            print("Error, if ydata is not given, explicitly specify that this is a histogram")
            
        elif ydata is None:
            #this is a histogram
            self.hist_data=xdata        
            hist, edges = np.histogram(xdata, bins=bins)
            self.hist_bins=edges
            _xdata = edges[:-1]
            _xerr = np.zeros(_xdata.size)
            _ydata = hist
            _yerr = np.sqrt(hist)        
        else:
            _xdata=xdata
            _xerr=xerr
            _ydata=ydata
            _yerr=yerr
        
        self.x = qe.MeasurementArray(_xdata,error=_xerr, name=xname, units=xunits)
        self.xdata = self.x.get_means()
        self.xerr = self.x.get_stds()
        self.xunits = self.x.get_units_str()
        self.xname = self.x.name
        
        self.y = qe.MeasurementArray(_ydata,error=_yerr, name=yname, units=yunits)
        self.ydata = self.y.get_means()
        self.yerr = self.y.get_stds()
        self.yunits = self.y.get_units_str()
        self.yname = self.y.name
        
        self.is_histogram=is_histogram
        
        if self.x.size != self.y.size:
            print("Error: x and y data should have the same number of points")
            #TODO raise an error!
        else:
            self.npoints = self.x.size
        
        self.xyfitter = []
        self.fit_pars = [] #stored as Measurement_Array
        self.fit_pcov = []
        self.fit_pcorr = []
        self.fit_function = [] 
        self.fit_function_name = []
        self.fit_npars =[]
        self.fit_yres = []
        self.fit_chi2 = []
        self.fit_ndof = []
        self.fit_color = []
        self.nfits=0
        
    def fit(self, model=None, parguess=None, fit_range=None, print_results=True, fitcolor=None):
        '''Fit a data set to a model using XYFitter. Everytime this function
        is called on a data set, it adds a new XYFitter to the dataset. This
        is to allow multiple functions to be fit to the same data set'''
        fitter = XYFitter(model=model, parguess=parguess)
        fit_pars = fitter.fit(self, fit_range=fit_range, fit_count=self.nfits)
        if(fit_pars is not None):
            self.xyfitter.append(fitter)
            self.fit_pars.append(fit_pars)
            self.fit_pcov.append(fitter.fit_pcov)
            self.fit_pcorr.append(cov2corr(fitter.fit_pcov))
            self.fit_npars.append(fit_pars.size)
            self.fit_yres.append(fitter.fit_yres)
            self.fit_function.append(fitter.fit_function) 
            self.fit_function_name.append(fitter.fit_function_name) 
            self.fit_chi2.append(fitter.fit_chi2)
            self.fit_ndof.append(fitter.fit_ndof)
            self.fit_color.append(fitcolor) # colour of the fit function
            self.nfits += 1
        
            if print_results:
                self.print_fit_results()
        
            return self.fit_pars[-1]
        else:
            return None
    
    def print_fit_results(self, fitindex=-1):
        if self.nfits == 0:
            print("no fit results to print")
            return
        print("-----------------Fit results-------------------")
        print("Fit of ",self.name," to ", self.fit_function_name[fitindex]) 
        print("Fit parameters:")
        print(self.fit_pars[fitindex])
        print("\nCorrelation matrix: ")
        print(np.array_str(self.fit_pcorr[fitindex], precision=3))
        print("\nchi2/ndof = {:.2f}/{}".format(self.fit_chi2[fitindex],self.fit_ndof[fitindex]))
        print("---------------End fit results----------------\n")
    
    def __str__(self):
        
        theString=""
        for i in range(self.xdata.size):
            theString += str(self.x[i])+" , "+str(self.y[i])+"\n"
        return theString
            
    def save_textfile(self, filename="dataset.dat", delim=' '):
        '''Save the data set to a file'''
        data = np.ndarray(shape=(self.xdata.size,4))
        data[:,0]=self.xdata
        data[:,1]=self.ydata
        data[:,2]=self.xerr
        data[:,3]=self.xerr
        np.savetxt(filename, data, fmt='%.4f', delimiter=delim)  
              
    def clear_fits(self):
        '''Remove all fit records'''
        self.xyfitter = []
        self.fit_pars = []
        self.fit_function = [] 
        self.fit_function_name = []
        self.fit_npars =[]
        self.yres = []
        self.nfits=0
    
    def get_x_range(self, margin=0):
        '''Get range of the x data, including errors and a specified margin'''
        if self.is_histogram:
            return [self.xdata.min()-margin,\
                    self.xdata.max()+margin]
        else:    
            return [self.xdata.min()-self.xerr.max()-margin,\
                    self.xdata.max()+self.xerr.max()+margin]
    
    def get_y_range(self, margin=0):
        '''Get range of the y data, including errors and a specified margin'''
        if self.is_histogram:
            return [self.ydata.min()-margin,\
                    self.ydata.max()+margin]
        else:
            return [self.ydata.min()-self.yerr.max()-margin,\
                    self.ydata.max()+self.yerr.max()+margin] 
    
    def get_yres_range(self, margin=0, fitindex=-1):
        '''Get range of the y residuals, including errors and a specified margin'''
        return [self.fit_yres[fitindex].get_means().min()-self.yerr.max()-margin,\
                self.fit_yres[fitindex].get_means().max()+self.yerr.max()+margin] 
    
def cov2corr(pcov):
    '''Return a correlation matrix given a covariance matrix'''
    sigmas = np.sqrt(np.diag(pcov))
    dim = sigmas.size
    pcorr = np.ndarray(shape=(dim,dim))
    
    for i in range(dim):
        for j in range(dim):
            pcorr[i][j]=pcov[i][j]
            if sigmas[i] == 0 or sigmas[j] == 0:
                pcorr[i][j] = 0.
            else:
                pcorr[i][j] /= (sigmas[i]*sigmas[j])
    return pcorr
           
    
    
def num_der(function, point, dx=1e-10):
    '''
    Returns the first order derivative of a function.
    Used in combining xerr and yerr. Used to include
    x errors in XYFitter
    '''
    import numpy as np
    point = np.array(point)
    return np.divide(function(point+dx)-function(point), dx)   