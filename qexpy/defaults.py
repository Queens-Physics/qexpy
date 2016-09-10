#Default parameters

import matplotlib.pyplot as plt
#Matplotlib needs to know the dpi to convert between
#actual size and pixels
screen_dpi = plt.gcf().get_dpi()
if screen_dpi == 0:
        screen_dpi = 100
        
import bokeh.palettes as bpal        
            
plotting_params = {
    "fig_x_px": 600, #x dimension of figures in pixels
    "fig_y_px": 400, #y dimension of figures in pixels
    "fig_title_ftsize": 11, #font size for figure titles
    "fig_xtitle_ftsize": 11, #font size for x axis label
    "fig_ytitle_ftsize": 11, #font size for y axis label
    "fig_fitres_ftsize": 11, #font size for fit results
    "fig_leg_ftsize": 11, #font size for legends
    "screen_dpi": screen_dpi, #default dpi used by matplotlib
    "color_palette": bpal.Set1_9+bpal.Set2_8, #color palette to choose colors from
    }
