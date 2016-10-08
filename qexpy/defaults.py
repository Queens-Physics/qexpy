#Default parameters

import matplotlib.pyplot as plt
#Matplotlib needs to know the dpi to convert between
#actual size and pixels
screen_dpi = plt.gcf().get_dpi()
if screen_dpi == 0:
        screen_dpi = 100
        
import bokeh.palettes as bpal        
   
settings = { 
    "plot_fig_x_px": 600, #x dimension of figures in pixels
    "plot_fig_y_px": 400, #y dimension of figures in pixels
    "plot_fig_title_ftsize": 11, #font size for figure titles
    "plot_fig_xtitle_ftsize": 11, #font size for x axis label
    "plot_fig_ytitle_ftsize": 11, #font size for y axis label
    "plot_fig_fitres_ftsize": 11, #font size for fit results
    "plot_fig_leg_ftsize": 11, #font size for legends
    "plot_screen_dpi": screen_dpi, #default dpi used by matplotlib
    "plot_color_palette": bpal.Set1_9+bpal.Set2_8, #color palette to choose colors from
    "plot_fcn_npoints": 100, #number of points to use for plotting functions and error bands
    "fit_max_fcn_calls": -1, #max number of function calls when fitting before giving up (-1 default)
    }   

