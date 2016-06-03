from scipy.optimize import curve_fit
import numpy as np
from uncertainties import Measured as M
from math import exp

from bokeh.plotting import figure, show
from bokeh.io import output_file, gridplot

ARRAY = (list, tuple,)

class Plot:
    '''
    Class of objects which can be plotted to display measurement objects
    which contain data and error data.
    '''
                   
    def polynomial(x,pars):
        poly=0
        n=0
        for par in pars:
            poly+=par*x**n
            n+=1
        return poly;

    fit={'linear':lambda x,pars: pars[0]+pars[1]*x,
         'exponential':lambda x,pars: exp(pars[0]+pars[1]*x),
         'polynomial':lambda x,pars: Plot.polynomial(x,pars),}
    
    def __init__(self,x,y,xerr=None,yerr=None):
        '''
        Object which can be plotted.   
        '''
        
        colors={'Data Points':'red','Function':'blue','Error':'red'}
        fit_method='linear'
        fit_function=Plot.fit[fit_method] #analysis:ignore
        plot_para={'xscale':'linear','yscale':'linear','filename':'Plot'}
        flags={'fitted':False,'residuals':False,} #analysis:ignore
        attributes={'title':'','xaxis':'x','yaxis':'y', #analysis:ignore
                            'data':'Experiment','functions':(),}
        
        data_transform(self,x,y,xerr,yerr)
        output_file(plot_para['filename'])
        
        # create a new plot
        p = figure(
            tools="pan,box_zoom,reset,save,wheel_zoom",width=600, height=400,
            y_axis_type=plot_para['filename'],
                        y_range=[min(self.ydata)-1.1*max(self.yerr),
                                 max(self.ydata)+1.1*max(self.yerr)],
            x_axis_type=plot_para['filename'],
                        x_range=[min(self.xdata)-1.1*max(self.xerr), 
                                 max(self.xdata)+1.1*max(self.xerr)],
            title=x.name+" versus "+y.name,
            x_axis_label=x.name+self.xunits, 
            y_axis_label=y.name+self.yunits
        )   
    
        # add datapoints with errorbars
        p.circle(self.xdata, self.ydata, legend=attributes['data'],
                                     color=colors['Data Points'], size=2) 
        error_bar(self,p)
        
    def residuals(self):
        
        if self.flag['fitted'] is False:
            Plot.fit(self.fit_function)
        
        p2 = figure(
            tools="pan,box_zoom,reset,save,wheel_zoom",width=600, height=200,
            y_axis_type='linear', y_range=[min(self.yres)-1.1*max(self.yerr),
                                           max(self.yres)+1.1*max(self.yerr)], 
            title="Residual Plot",
            x_axis_label=self.x_axis_label, 
            y_axis_label='Residuals'
        )
        
        #Calculate residual values
        yfit = self.fit.model(self.xdata,*self.fit.pars_fit)
        #Generate a set of residuals for the fit
        self.yres = self.ydata-yfit
        
        # add some renderers
        p2.circle(self.xdata, self.yres, color="black", size=2)
        
        # plot y errorbars
        error_bar(self,p2)
        
        self.flag['residuals']=True
    
    def fit(self,model=None):
        
        if model is not None:
            if type(model) is not str:
                self.fit_function=model
                self.flag['Unknown Function']=True
            else:
                self.fit_model(model)
        
        def model(x,*pars):
            return self.fit_funciton
        
        pars_guess = [1,1]

        pars_fit, pcov=curve_fit(model, self.xdata, self.ydata, 
                                 sigma=self.yerr, p0 = pars_guess)
        pars_err = np.sqrt(np.diag(pcov))
        
        slope=M(pars_fit[1],pars_err[1])
        slope.rename('Slope')
        intercept=M(pars_fit[0],pars_err[0])
        intercept.rename('Intercept')
        
        self.fit_parameters=(slope,intercept)
        
        self.flag['fitted']=True
        
    def function(self,function):
        self.attributes['function']+=function
        
    def plot(self):
        '''
        Method which creates and displays plot.
        Previous methods sumply edit parameters which are used here, to
        prevent run times increasing due to rebuilding the bokeh plot object.
        '''
        
        # create a new plot
        p = figure(
            tools="pan,box_zoom,reset,save,wheel_zoom",width=600, height=400,
            y_axis_type=self.plot_para['filename'],
                        y_range=[min(self.ydata)-1.1*max(self.yerr),
                                 max(self.ydata)+1.1*max(self.yerr)],
            x_axis_type=self.plot_para['filename'],
                        x_range=[min(self.xdata)-1.1*max(self.xerr), 
                                 max(self.xdata)+1.1*max(self.xerr)],
            title=self.attributes['xaxis']+" versus "+self.attributes['yaxis'],
            x_axis_label=self.attributes['xaxis']+self.xunits, 
            y_axis_label=self.attributes['yaxis']+self.yunits
        )   
    
        # add datapoints with errorbars
        p.circle(self.xdata, self.ydata, legend=self.attributes['data'],
                                     color=self.colors['Data Points'], size=2) 
        error_bar(p,self.xdata,self.ydata,self.xerr,self.yerr)
        
        for func in self.attributes['function']:
            _plot_function(p,self.xdata,func)
        
        if self.flag['fitted'] is True:
            _plot_function(p,self.xdata,self.fit_function)
            
            if self.fit_parameters[1].mean>0:
                p.legend.location = "top_left"  
            else:
                p.legend.location = "top_right" 
        else:
            p.legend.location = 'top_right'
        
        if self.flag['residuals'] is False:
            show(p)
        else:
            
            p2 = figure(
                tools="pan,box_zoom,reset,save,wheel_zoom",width=600, 
                                                                height=200,
                y_axis_type='linear', y_range=[min(self.yres)
                    -1.1*max(self.yerr),max(self.yres)+1.1*max(self.yerr)], 
                title="Residual Plot",
                x_axis_label=self.attributes['xaxis'], 
                y_axis_label='Residuals'
            )
            
            # add some renderers
            p2.circle(self.xdata, self.yres, color="black", size=2)
            
            # plot y errorbars
            error_bar(p2,self.xdata,self.yres,None,self.yerr)
            
            gp_alt = gridplot([[p],[p2]])
            show(gp_alt)
            
    def set_colors(self,data=None,error=None,line=None):
        if data is not None:
            self.colors['Data Points']=data
        if error is not None:
            self.colors['Error']=error
        if line is not None:
            self.colors['Function']=line
            
    def set_name(self,title=None,xname=None,yname=None,data_name=None,):
        if title is not None:
            self.attributes['title']=title
        if xname is not None:
            self.attributes['xname']=xname
        if yname is not None:
            self.attributes['yname']=yname
        if data_name is not None:
            self.attributes['Data']=data_name
            
    
        
def error_bar(self,p):
    # create the coordinates for the errorbars
    err_x1=[]
    err_d1=[]
    err_y1=[]
    err_d2=[]
    err_t1=[]
    err_t2=[]
    err_b1=[]
    err_b2=[]

    _xdata=self.xdata
    _ydata=self.ydata
    _yerr=self.yerr

    for _xdata, _ydata, _yerr in zip(_xdata, _ydata, _yerr):
        err_x1.append((_xdata, _xdata))
        err_d1.append((_ydata - _yerr, _ydata + _yerr)) 
        err_t1.append(_ydata+_yerr)
        err_b1.append(_ydata-_yerr)
    
    
    p.multi_line(err_x1, err_d1, color='red')
    p.rect(x=[*self.xdata,*self.xdata],y=[*err_t1,*err_b1],height=0.2,width=5,
               height_units='screen',width_units='screen',color='red')
    
    _xdata=self.xdata
    _ydata=self.ydata
    _xerr=self.xerr    
        
    for _ydata, _xdata, _xerr in zip(_ydata, _xdata, _xerr):
        err_y1.append((_ydata, _ydata))
        err_d2.append((_xdata - _xerr, _xdata + _xerr)) 
        err_t2.append(_xdata+_xerr)
        err_b2.append(_xdata-_xerr)
        
    p.multi_line(err_d2, err_y1, color='red')
    p.rect(x=[*err_t2,*err_b2],y=[*self.ydata,*self.ydata],height=5,width=0.2,
           height_units='screen',width_units='screen',
           color=self.colors['Data Points'])
               
def data_transform(self,x,y,xerr=None,yerr=None):  
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
    self.xdata=xdata
    self.ydata=ydata
    self.xerr=x_error
    self.yerr=y_error
    self.xunits=xunits
    self.yunits=yunits

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
                    