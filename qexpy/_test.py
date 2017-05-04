import math as m
import qexpy.error as e
import qexpy.plotting as p
import unittest
import numpy as np


class TestError(unittest.TestCase):
    def test_single_measurement(self):
        '''Tests creating a Measurement from a single measurement with uncertainty
        '''
        x = e.Measurement(10, 1)
        self.assertEqual(x.mean, 10)
        self.assertEqual(x.std, 1)
        x.mean = 3
        x.std = 0.1
        self.assertEqual(x.mean, 3)
        self.assertEqual(x.std, 0.1)

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

    def test_measurement_elementary(self):
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

    def test_array_elementary(self):
        '''Tests elementary operations on Measurement objects
        '''
        x = e.MeasurementArray([4, 9, 2], error = [0.1, 0.2, 0.3])
        y = e.MeasurementArray([2, 3, 3], error = 0.1)

        self.assertListEqual((x+y).means.tolist(), [6, 12, 5])
        self.assertListEqual((y+x).means.tolist(), [6, 12, 5])
        self.assertListEqual((x-y).means.tolist(), [2, 6, -1])
        self.assertListEqual((y-x).means.tolist(), [-2, -6, 1])
        self.assertListEqual((x*y).means.tolist(), [8, 27, 6])
        self.assertListEqual((y*x).means.tolist(), [8, 27, 6])
        self.assertListEqual((x/y).means.tolist(), [2, 3, 2/3])
        self.assertListEqual((y/x).means.tolist(), [0.5, 1/3, 3/2])
        self.assertListEqual((x**y).means.tolist(), [16, 729, 8])
        self.assertListEqual((y**x).means.tolist(), [16, 19683, 9])

    def test_measurement_functions(self):
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

    def test_measurement_comparisons(self):
        '''Tests comparisons of Measurement objects
        '''
        x = e.Measurement(3.2, 0.01)
        y = e.Measurement(0.23, 0.04)
        z = e.Measurement(0.23, 0.01)

        self.assertFalse(x == y)
        self.assertFalse(y == x)
        self.assertTrue(y == z)
        self.assertTrue(z == y)

        self.assertFalse(x <= y)
        self.assertFalse(y >= x)
        self.assertTrue(y >= z)
        self.assertTrue(z <= y)

        self.assertTrue(x > y)
        self.assertTrue(y < x)
        self.assertFalse(y > z)
        self.assertFalse(z < y)

    def test_array_functions(self):
        '''Tests mathematical functions on Measurement objects
        '''
        x = e.MeasurementArray([4, 9, 2], error = [0.1, 0.2, 0.3])
        y = e.MeasurementArray([0.3, 0.56, 0.2], error = 0.01)

        self.assertEqual(e.sin(x).means.tolist(), np.sin(x.means).tolist())
        self.assertEqual(e.cos(x).means.tolist(), np.cos(x.means).tolist())
        self.assertEqual(e.tan(x).means.tolist(), np.tan(x.means).tolist())
        self.assertEqual(e.csc(x).means.tolist(), (1/np.sin(x.means)).tolist())
        self.assertEqual(e.sec(x).means.tolist(), (1/np.cos(x.means)).tolist())
        self.assertEqual(e.cot(x).means.tolist(), (1/np.tan(x.means)).tolist())
        self.assertEqual(e.exp(x).means.tolist(), np.exp(x.means).tolist())
        self.assertEqual(e.log(x).means.tolist(), np.log(x.means).tolist())
        self.assertEqual(e.asin(y).means.tolist(), np.arcsin(y.means).tolist())
        self.assertEqual(e.acos(y).means.tolist(), np.arccos(y.means).tolist())
        self.assertEqual(e.atan(x).means.tolist(), np.arctan(x.means).tolist())

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
