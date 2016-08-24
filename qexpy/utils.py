
def in_notebook():
    '''Simple function to check if module is loaded in a notebook'''
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

#Global variable to keep track of whether the output_notebook command was run    
bokeh_ouput_notebook_called = False
    
    