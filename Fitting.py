import pylab as pl
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import numpy as np
from uncertainties import Measured as M

from bokeh.plotting import figure, show
from bokeh.io import output_file

ARRAY = (list,tuple,)

def linear_plot(x,y,xerr=None,yerr=None,title='Linear Plot', 
                parameters=['x','y','m','b']):
    
    if xerr is None:
        xdata=x.mean*np.linspace(1,len(y),len(y))
        xerr=x.std #*np.linspace(1,len(y),len(y))
    else:
        xdata=x
        xerr=xerr
    if yerr is None:
        ydata=y.mean*np.linspace(1,len(x),len(x))
        yerr=y.std #*np.linspace(1,len(x),len(x))
    else:
        ydata=y
        yerr=yerr
    
    try:
        x.units
    except AttributeError:
        xunits='unitless'
    else:
        if x.units is not '':
            xunits=x.units
        else:
            xunits='unitless'
    
    try:
        y.units
    except AttributeError:
        yunits='unitless'
    else:
        if y.units is not '':
            yunits=y.units
        else:
            yunits='unitless'
    
    def model(x,*pars):
        return pars[0]+pars[1]*x
    
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
    
    #Set the size of the figure
    pl.figure(figsize=(8,8))
    #Divide the figure into 2 rows, with 1 row 3 times higher for the data
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    pl.subplot(gs[0])
    #Plot the data with error bars
    pl.errorbar(xdata,ydata,yerr=yerr,xerr=xerr,fmt='o',
                                                label='data',color='black')
    #Plot the fit line
    pl.plot(xdata,yfit,'r',label='fit',lw=3)
    #Set the axes range to be pretty:
    pl.axis([xdata.min()-1,xdata.max()+1,ydata.min()-yerr[0]-1,
                                                             1.1*ydata.max()])
    pl.legend(loc='best')
    #Placement of the textbox may not be ideal:
    pl.text(xdata.mean()-10,ydata.min(),resultTxt,fontsize=14)
    pl.title(title)
    pl.xlabel(parameters[0]+'['+xunits+']')
    pl.ylabel(parameters[1]+'['+yunits+']')
    #Use the bottom row of the figure for the residuals:
    pl.subplot(gs[1])
    pl.errorbar(xdata,yres,yerr=yerr,fmt='o',color='black')#residuals
    pl.ylabel('residuals')
    pl.xlabel(parameters[0]+'['+xunits+']')
    pl.axis([xdata.min()-1,xdata.max()+1,-2*yerr.max(),2*yerr.max()])
    pl.tight_layout()
    pl.savefig(title+'.png')
    pl.show()
    return (slope,intercept,)
    
def bokehplot(x1, y1, d1):
    
    # output to notebook
    output_file('Plot')

    # create a new plot
    p = figure(
        tools="pan,box_zoom,reset,save", width=800, height=400,
        y_axis_type="linear", y_range=[8.0, 22], 
        title="Theory versus Experiment",x_axis_label='X (m)', 
        y_axis_label='Amplitude / V'
    )   

    # add some renderers
    p.line(x1, y1, legend="theory f(x)")
    p.circle(x1, d1, legend="experiment", fill_color="white", size=8)

    # create the coordinates for the errorbars
    err_x1 = []
    err_d1 = []

    yerrs = np.random.uniform(low = 1, high = 2.0, size = x1.shape)

    for x1, d1, yerr in zip(x1, d1, yerrs):
        err_x1.append((x1, x1))
        err_d1.append((d1 - yerr, d1 + yerr))

    # plot them
    p.multi_line(err_x1, err_d1, color='red')
    p.legend.location = "top_left"

    # show the results
    show(p)
