import math as m
import qexpy.error as e
import qexpy.plotting as p
import unittest


class TestError(unittest.TestCase):
    def test_single_measurement(self):
        '''Tests creating a Measurement from a single measurement with uncertainty
        '''
        x = e.Measurement(10, 1)
        self.assertEqual(x.mean, 10)
        self.assertEqual(x.std, 1)

    def test_multiple_measurements(self):
        '''Tests creating a Measurement from a multiple measurements
        '''
        x = e.Measurement([9, 10, 11])
        self.assertEqual(x.mean, 10)
        self.assertEqual(x.std, 1)

    def test_measurement_array(self):
        '''Tests creating a MeasurementArray from multiple measurements
        '''
        x = e.MeasurementArray([9, 10, 11], error=1)
        self.assertEqual(x.mean, 10)
        self.assertEqual(x.std(), 1)


class TestFunctions(unittest.TestCase):
    def test_elementary(self):
        '''Tests elementary operations on Measurement objects
        '''
        x = e.Measurement(2, 0.01)
        y = e.Measurement(5, 0.2)
        self.assertEqual(x+y, e.Measurement(7, 0.2))
        self.assertEqual(y+x, e.Measurement(7, 0.2))
        self.assertEqual(x-y, e.Measurement(-3, .2))
        self.assertEqual(y-x, e.Measurement(3, .2))
        self.assertEqual(x*y, e.Measurement(10, 0.4))
        self.assertEqual(y*x, e.Measurement(10, 0.4))
        self.assertEqual(x/y, e.Measurement(.4, .02))
        self.assertEqual(y/x, e.Measurement(2.5, 0.1))
        self.assertEqual(x**y, e.Measurement(32, 5))
        self.assertEqual(y**x, e.Measurement(25, 2))

    def test_functions(self):
        '''Tests mathematical functions on Measurement objects
        '''
        x = e.Measurement(3.2, 0.01)
        y = e.Measurement(0.23, 0.04)
        self.assertEqual(e.sin(x), m.sin(x.mean))
        self.assertEqual(e.cos(x), m.cos(x.mean))
        self.assertEqual(e.tan(x), m.tan(x.mean))
        self.assertEqual(e.csc(x), 1/m.sin(x.mean))
        self.assertEqual(e.sec(x), 1/m.cos(x.mean))
        self.assertEqual(e.cot(x), 1/m.tan(x.mean))
        self.assertEqual(e.exp(x), m.exp(x.mean))
        self.assertEqual(e.log(x), m.log(x.mean))
        self.assertEqual(e.asin(y), m.asin(y.mean))
        self.assertEqual(e.acos(y), m.acos(y.mean))
        self.assertEqual(e.atan(x), m.atan(x.mean))

    def test_derivative(self):
        '''Tests derivative of functions of Measurement objects
        '''
        x = e.Measurement(3, 0.4)
        y = e.Measurement(12, 1)

        self.assertEqual((x+y).get_derivative(y), 1)
        self.assertEqual((x-y).get_derivative(x), 1)
        self.assertEqual((x*y).get_derivative(y), x.mean)
        self.assertEqual((x/y).get_derivative(x), 1/y.mean)
        self.assertEqual((x**y).get_derivative(x), y.mean*x.mean**(y.mean-1))
        self.assertEqual(e.sin(x).get_derivative(x), m.cos(x.mean))
        self.assertEqual(e.cos(x).get_derivative(x), -m.sin(x.mean))
        self.assertEqual(e.tan(x).get_derivative(x), m.cos(x.mean)**-2)
        self.assertEqual(e.exp(x).get_derivative(x), m.exp(x.mean))

class TestFitting(unittest.TestCase):
    def test_linear_fit(self):
        ''' Test of plotting fit
        '''
        X = e.MeasurementArray([1, 2, 3, 4, 5], [0.1])
        Y = e.MeasurementArray([3, 5, 7, 9, 11], [0.05])

        figure = p.MakePlot(xdata = X, ydata = Y)
        figure.fit('linear')
        intercept = figure.get_dataset().xyfitter[0].fit_pars[0]
        slope = figure.get_dataset().xyfitter[0].fit_pars[1]

        self.assertEqual(slope, 2)
        self.assertEqual(intercept, 1)

# TODO: add tests of other fits

class TestMisc(unittest.TestCase):
    def test_unit_propagation(self):
        '''Tests unit propagation of Measurements
        '''
        L = e.Measurement(12, 1, name='Distance', units='m')
        v = e.Measurement(5, 0.1, name='Velocity', units=['m', 1, 's', -1])
        t = L/v
        self.assertEqual(L.units, {'m': 1})
        self.assertEqual(v.units, {'s': -1, 'm': 1})
        self.assertEqual(t.units, {'s': 1})

        x = e.Measurement(2, 0.3, name='Length', units='m')
        x2 = x + L
        self.assertEqual(x2.units, {'m': 1})

        L = v*t
        self.assertEqual(L.units, {'m': 1})


    def test_printing(self):
        '''Test of printing methods and sigfigs.
        '''
        # Test of standard printing without figs ##################################
        x = e.Measurement(12563.2, 1.637)
        e.set_print_style('Latex')
        self.assertEqual(x.__str__(), '(12563 \pm 2)')

        x = e.Measurement(156.2, 12)
        e.set_print_style('Default')
        self.assertEqual(x.__str__(), '160 +/- 10')

        x = e.Measurement(1360.2, 16.9)
        e.set_print_style('Sci')
        self.assertEqual(x.__str__(), '(136 +/- 2)*10^(1)')

        # Test of figs set on central value #######################################
        e.set_print_style('Default', 3)
        x = e.Measurement(12.3, 0.1, name='x')
        self.assertEqual(x.__str__(), 'x = 12.3 +/- 0.1')

        e.set_print_style('Latex', 4)
        x = e.Measurement(12.3, 0.156, name='x')
        self.assertEqual(x.__str__(), 'x = (1230 \pm 16)*10^{-2}')

        e.set_print_style('Sci', 5)
        x = e.Measurement(123.456, 0.789, name='x')
        self.assertEqual(x.__str__(), 'x = (12346 +/- 79)*10^(-2)')

        # Test of figs set on uncertainty #########################################
        x = e.Measurement(12.35, 0.1237)
        e.set_print_style('Default')
        e.set_sigfigs_error()
        self.assertEqual(x.__str__(), '12.350 +/- 0.124')

        x = e.Measurement(120, 0.1237795)
        e.set_print_style('Latex')
        e.set_sigfigs_error(5)
        self.assertEqual(x.__str__(), '(12000000 \pm 12378)*10^{-5}')

        x = e.Measurement(12.38, 0.1237)
        e.set_print_style('Sci')
        e.set_sigfigs_error(1)
        self.assertEqual(x.__str__(), '(124 +/- 1)*10^(-1)')


    def test_public_methods(self):
        '''Test of public methods to return Measurement object attributes.
        '''
        x = e.Measurement(10, 1, name='x', units='m')
        y = e.Measurement(13, 2, name='y', units=['m', 1])
        a = x+y

        d = e.Measurement(21, 1, name='Distance', units='m')
        t = e.Measurement(7, 2, name='Interval', units='s')
        v = d/t

        self.assertEqual(x.mean, 10)
        self.assertEqual(y.std, 2)
        self.assertEqual(a.get_derivative(x), 1)
        self.assertEqual(a.name, 'x+y')
        self.assertEqual(a.get_units(), 'm')
        self.assertTrue(v.get_units() == 'm^1 s^-1 ' or v.get_units() == 's^-1 m^1 ')
