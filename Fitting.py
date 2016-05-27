import pylab as pl
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import numpy as np

ARRAY = (list,tuple,)

def linear_plot(x,y,xerr=None,yerr=None):
    
    if xerr is None:
        xdata=x.mean*np.linspace(1,len(y),len(y))
        xerr=x.std#*np.linspace(1,len(y),len(y))
    else:
        xdata=x
        xerr=xerr
    if yerr is None:
        ydata=y.mean*np.linspace(1,len(x),len(x))
        yerr=y.std#*np.linspace(1,len(x),len(x))
    else:
        ydata=y
        yerr=yerr
    
    def model(x,*pars):
        return pars[0]+pars[1]*x

    pars_guess = [1,1]
    
    pars_fit, pcov=curve_fit(model, xdata, ydata, sigma=yerr, p0 = pars_guess)
    pars_err = np.sqrt(np.diag(pcov))
    
    #Create some text with the fit results to put into our plot
    resultTxt = '''Fitted parameters for:
    $\ln(A) =-\gamma /2t+b$:\n'''
    parNames = ["$b$","$-\gamma /2$"]
    for i in range(pars_fit.size):
        resultTxt = resultTxt+"{:s}: {:.2f} +/- {:.2f}\n".format(parNames[i],pars_fit[i],pars_err[i])
    
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
    pl.errorbar(xdata,ydata,yerr=yerr,xerr=xerr,fmt='o',label='data',color='black')
    #Plot the fit line
    pl.plot(xdata,yfit,'r',label='fit',lw=3)
    #Set the axes range to be pretty:
    pl.axis([xdata.min()-1,xdata.max()+1,ydata.min()-yerr[0]-1,1.1*ydata.max()])
    pl.legend(loc='best')
    #Placement of the textbox may not be ideal:
    pl.text(xdata.mean()-10,ydata.min(),resultTxt,fontsize=14)
    pl.title("Underdamped Mechanical Oscillator Fit")
    pl.xlabel('t [s]')
    pl.ylabel('log(|A|)')
    #Use the bottom row of the figure for the residuals:
    pl.subplot(gs[1])
    pl.errorbar(xdata,yres,yerr=yerr,fmt='o',color='black')#residuals
    pl.ylabel('residuals')
    pl.xlabel('t [s]')
    pl.axis([xdata.min()-1,xdata.max()+1,-2*yerr.max(),2*yerr.max()])
    pl.tight_layout()
    pl.savefig("MechUnder.png")
    pl.show()