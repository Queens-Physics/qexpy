import error as e
import plotting as p

x = e.Measurement([1, 2, 3, 4, 5], [0.5], name='Length', units='cm')
y = e.Measurement([5, 7, 11, 14, 17], [1], name='Appplied Mass', units='g')

figure = p.Plot(x, y)
figure.fit('linear')
figure.residuals()
figure.show('file')
