#import pylab as pl
from scipy.optimize import curve_fit
#import matplotlib.gridspec as gridspec
import numpy as np
from uncertainties import Measured as M

from bokeh.plotting import figure, show
from bokeh.io import output_file,gridplot

ARRAY = (list,tuple,)

def error_bar(xdata,ydata,yerr):
    # create the coordinates for the errorbars
    err_x1 = []
    err_d1 = []

    _xdata=xdata
    _ydata=ydata
    _yerr=yerr

    for _xdata, _ydata, _yerr in zip(_xdata, _ydata, _yerr):
        err_x1.append((_xdata, _xdata))
        err_d1.append((_ydata - _yerr, _ydata + _yerr)) 
    
    return (err_x1,err_d1,)

def data_transform(x,y,xerr,yerr):
    if xerr is None:
        xdata=x.mean*np.linspace(1,len(y.info['Data']),len(y.info['Data']))
        x_error=[x.std]*len(y.info['Data']) #*np.linspace(1,len(y),len(y))
    else:
        try:
            x.type
        except AttributeError:
            xdata=x
            x_error=xerr
        else:
            xdata=x.info['Data']
            x_error=xerr
        
    if yerr is None:
        ydata=y.mean*np.linspace(1,len(x),len(x))
        y_error=[y.std]*len(x.info['Data']) #*np.linspace(1,len(x),len(x))
    else:
        try:
            y.type
        except AttributeError:
            ydata=y
            y_error=yerr
        else:
            ydata=y.info['Data']
            y_error=yerr
    
    try:
        x.units
    except AttributeError:
        xunits='unitless'
    else:
        if x.units is not '':
            xunits=''
            for key in x.units:
                xunits+=key+'^%d'%(x.units[key])
        else:
            xunits='unitless'
    
    try:
        y.units
    except AttributeError:
        yunits='unitless'
    else:
        if y.units is not '':
            yunits=''
            for key in y.units:
                yunits+=key+'^%d'%(y.units[key])
        else:
            yunits='unitless'   

    return (xdata,ydata,x_error,y_error,xunits,yunits,); 

def theoretical_plot(theory,x,y,xerr=None,yerr=None,
                             parameters=['x','y','m','b'],filename='Plot'):
                                             
    xdata,ydata,xerr,yerr,xunits,yunits=data_transform(x,y,xerr,yerr)
    
    output_file(filename)
    
    # create a new plot
    p = figure(
        tools="pan,box_zoom,reset,save,wheel_zoom", 
                                                        width=600, height=400,
        y_axis_type='linear', y_range=[min(ydata)-1.1*max(yerr), 
                                                    max(ydata)+1.1*max(yerr)],
        x_axis_type='linear', x_range=[min(xdata)-1.1*max(xerr), 
                                                    max(xdata)+1.1*max(xerr)], 
        title=x.name+" versus "+y.name,
        x_axis_label=parameters[0]+'['+xunits+']', 
        y_axis_label=parameters[1]+'['+yunits+']'
    )   

    # add some renderers
    p.circle(xdata, ydata, legend="experiment", color="black", size=2) 
    
    #Plot theoretical line
    xrange=np.linspace(min(xdata),max(xdata),1000)
    p.line(xrange,theory(xrange),legend='Theoretical')

    err_x1,err_d1=error_bar(xdata,ydata,yerr)
    err_y1,err_d2=error_bar(ydata,xdata,xerr)
    
    # plot them
    p.multi_line(err_x1, err_d1, color='red')
    p.multi_line(err_d2, err_y1, color='red')
    p.legend.location = "top_right"     
    
    #plotting and returning slope and intercept
    show(p)

def fitted_plot(x,y,xerr=None,yerr=None,parameters=['x','y','m','b'],
                                    fit='linear',theory=None,filename='Plot'):
    from numpy import exp

    xdata,ydata,xerr,yerr,xunits,yunits=data_transform(x,y,xerr,yerr)
    
    def model(x,*pars):
        from numpy import multiply as m
        fits={'linear':pars[0]+m(pars[1],x),
              'log':exp(pars[0]+m(pars[1],x)),}
        return fits[fit]
    
    pars_guess = [1,1]

    pars_fit, pcov=curve_fit(model, xdata, ydata, sigma=yerr, p0 = pars_guess)
    pars_err = np.sqrt(np.diag(pcov))
    
    slope=M(pars_fit[1],pars_err[1])
    slope.rename('Slope')
    intercept=M(pars_fit[0],pars_err[0])
    intercept.rename('Intercept')
    
    #Create some text with the fit results to put into our plot
    resultTxt = "Model: $"\
        +parameters[1]+'='+parameters[2]+parameters[0]+\
                                    '+'+parameters[3]+'$:\n'
     
    parNames = [parameters[3],parameters[2]]
    for i in range(pars_fit.size):
        resultTxt = resultTxt+"${:s}: {:.2f} +/- {:.2f}$\n".format(
                                        parNames[i],pars_fit[i],pars_err[i])
    
    #########################################################    
    #Plot the data with error bars and the result of the fit#
    #Also include a subplot with the residuals              #
    #########################################################
    #Generage a curve from the model and the fitted parameters
    yfit = model(xdata,*pars_fit)
    #Generate a set of residuals for the fit
    yres = ydata-yfit
    
    # output to notebook
    output_file(filename)
    
    # create a new plot
    p = figure(
        tools="pan,box_zoom,reset,save,wheel_zoom", 
                                                        width=600, height=400,
        y_axis_type=fit, y_range=[min(ydata)-1.1*max(yerr), 
                                                    max(ydata)+1.1*max(yerr)],

        x_axis_type='linear', x_range=[min(xdata)-1.1*max(xerr), 
                                                    max(xdata)+1.1*max(xerr)], 
        title=x.name+" versus "+y.name,
        x_axis_label=parameters[0]+'['+xunits+']', 
        y_axis_label=parameters[1]+'['+yunits+']'
    )   

    # add some renderers
    p.line(xdata, yfit, legend="fit")
    p.circle(xdata, ydata, legend="experiment", color="black", size=2) 
    
    #Plot theoretical line if applicable
    if theory is not None:
        xrange=np.linspace(min(xdata),max(xdata),1000)
        p.line(xrange,theory(xrange),legend='Theoretical')
        
    err_x1,err_d1=error_bar(xdata,ydata,yerr)
    err_y1,err_d2=error_bar(ydata,xdata,xerr)
    
    # plot them
    p.multi_line(err_x1, err_d1, color='red')
    p.multi_line(err_d2, err_y1, color='red')
    if pars_fit[1]>0:
        p.legend.location = "top_left"  
    else:
        p.legend.location = "top_right"  
       
    
    # create a new plot
    p2 = figure(
        tools="pan,box_zoom,reset,save,wheel_zoom", 
                                                        width=800, height=200,
        y_axis_type='linear', y_range=[min(yres)-1.1*max(yerr),
                                                   max(yres)+1.1*max(yerr)], 
        title="Residual Plot",
        x_axis_label=parameters[0]+'['+xunits+']', 
        y_axis_label='Residuals'
    )
    
    # add some renderers
    p2.circle(xdata, yres, color="black", size=2)
    
    err_x1,err_d1=error_bar(xdata,yres,yerr)
    
    # plot them
    p2.multi_line(err_x1, err_d1, color='red')
    
    #plotting and returning slope and intercept
    gp_alt = gridplot([[p],[p2]])
    show(gp_alt)
    return (slope,intercept,)
