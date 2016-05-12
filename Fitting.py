import pylab as pl
ARRAY = (list,tuple,)

def plot(x,y):
    '''
    Plot measurements x and y
    
    Supported types are array, list of integers, floats, or measurements.
    '''
    if type(x) in ARRAY:
        pass
        #use pylab to plot, use std error if not array of measurements
