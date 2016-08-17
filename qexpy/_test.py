import math as m
import error as e
import plotting as p


def test1():
    '''Test of data entry methods.
    '''
    x = e.Measurement(10, 1)
    assert x.mean == 10
    assert x.std == 1

    x = e.Measurement([9, 10, 11], [1])
    assert x.mean == 10

    x = e.Measurement(9, 10, 11)
    assert x.mean == 10
    assert x.std == 1

    x = e.Measurement([9, 1], [10, 1], [11, 1])
    assert x.mean == 10


def test2():
    '''Test of elementary functions.
    '''
    x = e.Measurement(10, 1)
    y = e.Measurement(0, 0.1)

    assert e.sin(x) == m.sin(x.mean)
    assert e.cos(x) == m.cos(x.mean)
    assert e.tan(x) == m.tan(x.mean)
    assert e.csc(x) == 1/m.sin(x.mean)
    assert e.sec(x) == 1/m.cos(x.mean)
    assert e.cot(x) == 1/m.tan(x.mean)
    assert e.exp(x) == m.exp(x.mean)
    assert e.log(x) == m.log(x.mean)
    assert e.asin(y) == m.asin(y.mean)
    assert e.acos(y) == m.acos(y.mean)
    assert e.atan(x) == m.atan(x.mean)


def test3():
    '''Test of elementary operators.
    '''
    x = e.Measurement(10, 1)
    y = e.Measurement(20, 2)

    assert x+y == x.mean+y.mean
    assert x-y == x.mean-y.mean
    assert x*y == x.mean*y.mean
    assert x/y == x.mean/y.mean
    assert x**y == x.mean**y.mean


def test4():
    '''Test of derivative method.
    '''
    x = e.Measurement(3, 0.4)
    y = e.Measurement(12, 1)

    assert (x+y).return_derivative(y) == 1
    assert (x-y).return_derivative(x) == 1
    assert (x*y).return_derivative(y) == x.mean
    assert (x/y).return_derivative(x) == 1/y.mean
    assert (x**y).return_derivative(x) == y.mean*x.mean**(y.mean-1)
    assert e.sin(x).return_derivative(x) == m.cos(x.mean)
    assert e.cos(x).return_derivative(x) == -m.sin(x.mean)
    assert e.tan(x).return_derivative(x) == m.cos(x.mean)**-2
    assert e.exp(x).return_derivative(x) == m.exp(x.mean)


def test5():
    ''' Test of plotting fit
    '''
    X = e.Measurement_Array([1, 2, 3, 4, 5], [0.1])
    Y = e.Measurement_Array([3, 5, 7, 9, 11], [0.05])

    figure = p.Plot(X, Y)
    figure.fit('linear')
    intercept, slope = figure.fit_parameters

    assert slope.mean == 2
    assert intercept.mean == 1


def test6():
    '''Test naming and unit propagation
    '''
    L = e.Measurement(12, 1, name='Distance', units='m')
    v = e.Measurement(5, 0.1, name='Velocity', units=['m', 1, 's', -1])
    t = L/v
    assert L.units == {'m': 1}
    assert v.units == {'s': -1, 'm': 1}
    assert t.units == {'s': 1}

    x = e.Measurement(2, 0.3, name='Length', units='m')
    x2 = x + L
    assert x2.units == {'m': 1}

    L = v*t
    assert L.units == {'m': 1}
