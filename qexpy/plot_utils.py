import numpy as np
import qexpy.error as qe
import qexpy.utils as qu

CONSTANT = qu.number_types
ARRAY = qu.array_types

def bk_plot_dataset(figure, dataset, residual=False, color='black', fit_index=-1):
    '''Given a bokeh figure, this will add data points with errors from a dataset'''
  
    xdata = dataset.xdata
    xerr = dataset.xerr
    data_name = dataset.name
    
    index = fit_index if fit_index < dataset.nfits else -1.
    
    if residual is True and dataset.nfits>0:
        ydata = dataset.fit_yres[index].get_means()
        yerr = dataset.fit_yres[index].get_stds()
        data_name=None
    else:
        ydata = dataset.ydata
        yerr = dataset.yerr
     
    bk_add_points_with_error_bars(figure, xdata, ydata, xerr, yerr, color, data_name)
    
def bk_add_points_with_error_bars(figure, xdata, ydata, xerr=None, yerr=None, color='black', data_name='dataset'):
    '''Add data points to a bokeh plot. If the errors are given as numbers, 
    the same error bar is assume for all data points'''
    
    
    _xdata, _ydata, _xerr, _yerr = make_np_arrays(xdata,ydata,xerr,yerr)  
    
    if _xdata.size != _ydata.size:
        print("Error: x and y data must have the same number of points")
        return None 
    
    #Draw points:    
    figure.circle(_xdata, _ydata, color=color, size=4, legend=data_name)

    if isinstance(_xerr,np.ndarray) or isinstance(_yerr,np.ndarray):
        #Add error bars
        for i in range(_xdata.size):
            
            xcentral = [_xdata[i], _xdata[i]]
            ycentral = [_ydata[i], _ydata[i]]
            
            #x error bar, if the xerr argument was not none
            if xerr is not None:
                xends = []
                if _xerr.size == _xdata.size and _xerr[i]>0:
                    xends = [_xdata[i]-_xerr[i], _xdata[i]+_xerr[i]]
                elif _xerr.size == 1 and _xerr[0]>0:
                    xends = [_xdata[i]-_xerr[0], _xdata[i]+_xerr[0]]
                else:                    
                    pass
                
                if len(xends)>0:                    
                    figure.line(xends,ycentral, color=color)
                    #winglets on x error bar:
                    figure.rect(x=xends, y=ycentral, height=5, width=0.2,
                        height_units='screen', width_units='screen',
                        color=color)
                
            #y error bar    
            if yerr is not None:
                yends=[]
                if _yerr.size == _ydata.size and _yerr[i]>0:
                    yends = [_ydata[i]-_yerr[i], _ydata[i]+_yerr[i]]
                elif _yerr.size == 1 and _yerr[i]>0:
                    yends = [_ydata[i]-_yerr[0], _ydata[i]+_yerr[0]]
                else:
                    pass
                if len(yends)>0:          
                    figure.line(xcentral, yends, color=color)
                    #winglets on y error bar:
                    figure.rect(x=xcentral, y=yends, height=0.2, width=5,
                        height_units='screen', width_units='screen',
                        color=color)
                
def make_np_arrays(*args):
    '''Return a tuple where all of the arguments have been converted into 
    numpy arrays'''
    np_tuple=()
    for arg in args:
        if isinstance(arg,np.ndarray):
            np_tuple = np_tuple +(arg,)
        elif isinstance(arg, list):
            np_tuple = np_tuple +(np.array(arg),)
        elif isinstance(arg, qu.number_types):
            np_tuple = np_tuple +(np.array([arg]),)
        else:
            np_tuple = np_tuple +(None,)
    return np_tuple
    
def bk_plot_function(figure, function, xdata, pars=None, n=100, legend_name=None, color='black', errorbandfactor=1.0):
    '''Plot a function evaluated over the range of xdata - xdata only needs 2 values
    The function can be either f(x) or f(x, *pars). In the later case, if pars is
    a Measurement_Array (e.g. the parameters from a fit), then an error band is also
    added to the plot, corresponding to varying the parameters within their uncertainty.
    The errorbandfactor can be used to choose an error band that is larger than 1 standard
    deviation.
    '''
    
    xvals = np.linspace(min(xdata), max(xdata), n)
    
    if pars is None:
        fvals = function(xvals)
    elif isinstance(pars, qe.Measurement_Array):
        #TODO see if this can be sped up more
        recall = qe.Measurement.minmax_n
        qe.Measurement.minmax_n=1
        fmes = function(xvals, *(pars))
        fvals = fmes.get_means()
        qe.Measurement.minmax_n=recall
    elif isinstance(pars,(list, np.ndarray)):
        fvals = function(xvals, *pars)
    else:
        print("Error: unrecognized parameters for function")
        pass
    line = figure.line(xvals, fvals, legend=legend_name, line_color=color)
    
    #Add error band
    if isinstance(pars, qe.Measurement_Array):
        ymax = fmes.get_means()+errorbandfactor*fmes.get_stds()
        ymin = fmes.get_means()-errorbandfactor*fmes.get_stds()

        patch = figure.patch(x=np.append(xvals,xvals[::-1]),y=np.append(ymax,ymin[::-1]),
                     fill_alpha=0.3,
                     fill_color=color,
                     line_alpha=0.0,
                     legend=legend_name)
       
        return line, patch
    else:
        return line, None

