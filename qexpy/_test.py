import math as m
import qexpy.error as e
import qexpy.plotting as p


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

    assert (x+y).get_derivative(y) == 1
    assert (x-y).get_derivative(x) == 1
    assert (x*y).get_derivative(y) == x.mean
    assert (x/y).get_derivative(x) == 1/y.mean
    assert (x**y).get_derivative(x) == y.mean*x.mean**(y.mean-1)
    assert e.sin(x).get_derivative(x) == m.cos(x.mean)
    assert e.cos(x).get_derivative(x) == -m.sin(x.mean)
    assert e.tan(x).get_derivative(x) == m.cos(x.mean)**-2
    assert e.exp(x).get_derivative(x) == m.exp(x.mean)


def test5():
    ''' Test of plotting fit
    '''
    X = e.MeasurementArray([1, 2, 3, 4, 5], [0.1])
    Y = e.MeasurementArray([3, 5, 7, 9, 11], [0.05])

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


def test7():
    '''Test of printing methods and sigfigs.
    '''
    # Test of standard printing without figs ##################################
    x = e.Measurement(12563.2, 1.637)
    e.set_print_style('Latex')
    assert x.__str__() == '(12563 \pm 2)'

    x = e.Measurement(156.2, 12)
    e.set_print_style('Default')
    assert x.__str__() == '160 +/- 10'

    x = e.Measurement(1360.2, 16.9)
    e.set_print_style('Sci')
    assert x.__str__() == '(136 +/- 2)*10^(1)'

    # Test of figs set on central value #######################################
    e.set_print_style('Default', 3)
    x = e.Measurement(12.3, 0.1, name='x')
    assert x.__str__() == 'x = 12.3 +/- 0.1'

    e.set_print_style('Latex', 4)
    x = e.Measurement(12.3, 0.156, name='x')
    assert x.__str__() == 'x = (1230 \pm 16)*10^{-2}'

    e.set_print_style('Sci', 5)
    x = e.Measurement(123.456, 0.789, name='x')
    assert x.__str__() == 'x = (12346 +/- 79)*10^(-2)'

    # Test of figs set on uncertainty #########################################
    x = e.Measurement(12.35, 0.1237)
    e.set_print_style('Default')
    e.set_sigfigs_error()
    assert x.__str__() == '12.350 +/- 0.124'

    x = e.Measurement(120, 0.1237795)
    e.set_print_style('Latex')
    e.set_sigfigs_error(5)
    assert x.__str__() == '(12000000 \pm 12378)*10^{-5}'

    x = e.Measurement(12.38, 0.1237)
    e.set_print_style('Sci')
    e.set_sigfigs_error(1)
    assert x.__str__() == '(124 +/- 1)*10^(-1)'


def test8():
    '''Test of public methods to return Measurement object attributes.
    '''
    x = e.Measurement(10, 1, name='x', units='m')
    y = e.Measurement(13, 2, name='y', units=['m', 1])
    a = x+y

    d = e.Measurement(21, 1, name='Distance', units='m')
    t = e.Measurement(7, 2, name='Interval', units='s')
    v = d/t

    assert x.get_mean() == 10
    assert y.get_error() == 2
    assert a.get_derivative(x) == 1
    assert a.get_name() == 'x+y'
    assert a.get_units() == 'm'
    assert v.get_units() == 'm^1 s^-1 ' or v.get_units() == 's^-1 m^1 '
