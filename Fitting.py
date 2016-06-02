from scipy.optimize import curve_fit
import numpy as np
from uncertainties import Measured as M

from bokeh.plotting import figure, show
from bokeh.io import output_file,gridplot

ARRAY = (list,tuple,)

def error_bar(p,xdata,ydata,xerr,yerr):
    # create the coordinates for the errorbars
    err_x1=[]
    err_d1=[]
    err_y1=[]
    err_d2=[]
    err_t1=[]
    err_t2=[]
    err_b1=[]
    err_b2=[]

    _xdata=xdata
    _ydata=ydata
    _yerr=yerr

    for _xdata, _ydata, _yerr in zip(_xdata, _ydata, _yerr):
        err_x1.append((_xdata, _xdata))
        err_d1.append((_ydata - _yerr, _ydata + _yerr)) 
        err_t1.append(_ydata+_yerr)
        err_b1.append(_ydata-_yerr)
    
    
    p.multi_line(err_x1, err_d1, color='red')
    p.rect(x=[*xdata,*xdata],y=[*err_t1,*err_b1],height=0.2,width=5,
               height_units='screen',width_units='screen',color='red')
    
    if xerr is not None:
        _xdata=xdata
        _ydata=ydata
        _xerr=xerr    
        
        for _ydata, _xdata, _xerr in zip(_ydata, _xdata, _xerr):
            err_y1.append((_ydata, _ydata))
            err_d2.append((_xdata - _xerr, _xdata + _xerr)) 
            err_t2.append(_xdata+_xerr)
            err_b2.append(_xdata-_xerr)
            
        p.multi_line(err_d2, err_y1, color='red')
        p.rect(x=[*err_t2,*err_b2],y=[*ydata,*ydata],height=5,width=0.2,
               height_units='screen',width_units='screen',color='red')
    
def data_transform(x,y,xerr,yerr):  
    if xerr is None:
        xdata=x.info['Data']
        x_error=x.info['Error']
    else:
        try:
            x.type
        except AttributeError:
            xdata=x
        else:
            xdata=x.info['Data']
        if type(xerr) in (int,float,):
            x_error=[xerr]*len(xdata)
        else:
            x_error=xerr
        
    if yerr is None:
        ydata=y.info['Data']
        y_error=y.info['Error']
    else:
        try:
            y.type
        except AttributeError:
            ydata=y
        else:
            ydata=y.info['Data']
        if type(yerr) in (int,float,):
            y_error=[yerr]*len(ydata)
        else:
            y_error=yerr
    try:
        x.units
    except AttributeError:
        xunits=''
    else:
        if len(x.units) is not 0:
            xunits=''
            for key in x.units:
                xunits+=key+'^%d'%(x.units[key])
            xunits='['+xunits+']'
        else:
            xunits=''
    
    try:
        y.units
    except AttributeError:
        yunits=''
    else:
        if len(y.units) is not 0:
            yunits=''
            for key in y.units:
                yunits+=key+'^%d'%(y.units[key])
            yunits='['+yunits+']'
        else:
            yunits=''   

    return (xdata,ydata,x_error,y_error,xunits,yunits,); 

def theoretical_plot(theory,x,y,xerr=None,yerr=None,xscale='linear',
                yscale='linear',parameters=['x','y','m','b'],filename='Plot'):
                                             
    xdata,ydata,xerr,yerr,xunits,yunits=data_transform(x,y,xerr,yerr)
    
    output_file(filename)
    
    # create a new plot
    p = figure(
        tools="pan,box_zoom,reset,save,wheel_zoom", 
                                                        width=600, height=400,
        y_axis_type=yscale, y_range=[min(ydata)-1.1*max(yerr), 
                                                    max(ydata)+1.1*max(yerr)],
        x_axis_type=xscale, x_range=[min(xdata)-1.1*max(xerr), 
                                                    max(xdata)+1.1*max(xerr)],
        title=x.name+" versus "+y.name,
        x_axis_label=parameters[0]+xunits, 
        y_axis_label=parameters[1]+yunits
    )   

    # add datapoints with errorbars
    p.circle(xdata, ydata, legend="experiment", color="black", size=2) 
    error_bar(p,xdata,ydata,xerr,yerr)
    
    #Plot theoretical line
    _plot_function(p,xdata,theory)

        
    # plot them
    p.legend.location = "top_right"     
    
    #plotting and returning slope and intercept
    show(p)

def _plot_function(p,xdata,theory,n=1000):
    n=1000
    xrange=np.linspace(min(xdata),max(xdata),n)
    x_theory=theory(min(xdata))
    x_mid=[]
    try:
        x_theory.type
    except AttributeError:
        for i in range(n):
            x_mid.append(theory(xrange[i]))
        p.line(xrange,x_mid,legend='Theoretical')
    else:
        x_max=[]
        x_min=[]
        for i in range(n):
            x_theory=theory(xrange[i])
            x_mid.append(x_theory.mean)
            x_max.append(x_theory.mean+x_theory.std)
            x_min.append(x_theory.mean-x_theory.std)
        p.line(xrange,x_mid,legend='Theoretical',line_color='red')
        
        xrange_reverse=list(reversed(xrange))
        x_min_reverse=list(reversed(x_min))
        p.patch(x=[*xrange,*xrange_reverse],y=[*x_max,*x_min_reverse],
                    fill_alpha=0.3,fill_color='red',line_color='red',
                    line_dash='dashed',line_alpha=0.3)
        
def fitted_plot(x,y,xerr=None,yerr=None,parameters=['x','y','m','b'],
                                    fit='linear',theory=None,filename='Plot'):
    from numpy import exp

    xdata,ydata,xerr,yerr,xunits,yunits=data_transform(x,y,xerr,yerr)
    
    def model(x,*pars):
        from numpy import multiply as m
        #fits={'linear':pars[0]+m(pars[1],x),
        #      'exponential':exp(pars[0]+m(pars[1],x)),}
        return pars[0]+m(pars[1],x)
    
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

        x_axis_type='linear',x_range=[min(xdata)-1.1*max(xerr),
                                                     max(xdata)+1.1*max(xerr)],
        title=x.name+" versus "+y.name,
        x_axis_label=parameters[0]+xunits, 
        y_axis_label=parameters[1]+yunits
    )   

    # add some renderers
    p.line(xdata, yfit, legend="fit")
    p.circle(xdata, ydata, legend="experiment", color="black", size=2) 
    _plot_function(p,xdata,lambda x:model(x,intercept,slope))
    
    #Plot theoretical line if applicable
    if theory is not None:
        _plot_function(p,xdata,theory)
        
    # plot errorbars
    error_bar(p,xdata,ydata,xerr,yerr)
    if pars_fit[1]>0:
        p.legend.location = "top_left"  
    else:
        p.legend.location = "top_right"  
       
    
    # create a new plot
    p2 = figure(
        tools="pan,box_zoom,reset,save,wheel_zoom", 
                                                        width=600, height=200,
        y_axis_type='linear', y_range=[min(yres)-1.1*max(yerr),
                                                   max(yres)+1.1*max(yerr)], 
        title="Residual Plot",
        x_axis_label=parameters[0]+'['+xunits+']', 
        y_axis_label='Residuals'
    )
    
    # add some renderers
    p2.circle(xdata, yres, color="black", size=2)
    
    # plot y errorbars
    error_bar(p2,xdata,yres,None,yerr)    
    
    #plotting and returning slope and intercept
    gp_alt = gridplot([[p],[p2]])
    show(gp_alt)
    return (slope,intercept,)
