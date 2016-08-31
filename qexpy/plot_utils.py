import scipy.optimize as sp
import numpy as np
import qexpy.error as qe
import qexpy.utils as qu
import qexpy.fitting as qf

from math import pi
import bokeh.plotting as bp
import bokeh.io as bi
import bokeh.models as mo

CONSTANT = qu.number_types
ARRAY = qu.array_types

def plot_dataset(figure, dataset, residual=False, data_color='black'):
    '''Given a bokeh figure, this will add data points with errors from the dataset'''
  
    xdata = dataset.xdata
    xerr = dataset.xerr
    data_name = dataset.name
    
    if residual is True and dataset.nfits>0:
        ydata = dataset.yres[-1].get_means()
        yerr = dataset.yres[-1].get_stds()
        data_name=None
    else:
        ydata = dataset.ydata
        yerr = dataset.yerr
     
    add_points_with_error_bars(figure, xdata, ydata, xerr, yerr, data_color, data_name)
    
def add_points_with_error_bars(figure, xdata, ydata, xerr=None, yerr=None, data_color='black', data_name='dataset'):
    
    if xdata.size != ydata.size:
        print("Error: x and y data must have the same number of points")
        return None
    
    #Draw points:    
    figure.circle(xdata, ydata, color=data_color, size=2, legend=data_name)

    if isinstance(xerr,np.ndarray) or isinstance(yerr,np.ndarray):
        #Add error bars
        for i in range(xdata.size):
            
            xcentral = [xdata[i], xdata[i]]
            ycentral = [ydata[i], ydata[i]]
            
            #x error bar
            if isinstance(xerr,np.ndarray) and xerr.size == xdata.size and xerr[i]>0:
                xends = [xdata[i]-xerr[i], xdata[i]+xerr[i]]
                figure.line(xends,ycentral, color=data_color)
                #winglets on x error bar:
                figure.rect(x=xends, y=ycentral, height=5, width=0.2,
                    height_units='screen', width_units='screen',
                    color=data_color,legend=data_name)
                
            #y error bar    
            if isinstance(yerr,np.ndarray) and yerr.size == xdata.size and yerr[i]>0:    
                yends = [ydata[i]-yerr[i], ydata[i]+yerr[i]]
                figure.line(xcentral, yends, color=data_color)
                #winglets on y error bar:
                figure.rect(x=xcentral, y=yends, height=0.2, width=5,
                    height_units='screen', width_units='screen',
                    color=data_color,legend=data_name)
    
    
def plot_function(figure, function, xdata, fpars=None, n=100, legend_name=None, color='black', errorbandfactor=1.0):
    '''Plot a function evaluated over the range of xdata'''
    xvals = np.linspace(min(xdata), max(xdata), n)
    
    if fpars is None:
        fvals = function(xvals)
    elif isinstance(fpars, qe.Measurement_Array):
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1
        fmes = function(xvals, *(fpars))
        fvals = fmes.get_means()
        qe.Measurement.minmax_n=recall
    elif isinstance(fpars,(list, np.ndarray)):
        fvals = function(xvals, *fpars)
    else:
        print("Error: unrecognized parameters for function")
        pass
    figure.line(xvals, fvals, legend=legend_name, line_color=color)
    
    #Add error band
    if isinstance(fpars, qe.Measurement_Array):
        ymax = fmes.get_means()+errorbandfactor*fmes.get_stds()
        ymin = fmes.get_means()-errorbandfactor*fmes.get_stds()

        figure.patch(x=np.append(xvals,xvals[::-1]),y=np.append(ymax,ymin[::-1]),
                     fill_alpha=0.3,
                     fill_color=color,
                     line_alpha=0.0,
                     legend=legend_name)
       
        

