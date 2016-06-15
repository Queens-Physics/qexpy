import uncertainties as u
import fitting as fit

# This cell will load the package in a way that is shorter to type

# We can now enter the data gathered in the lab itself

dl = u.Measured([185e-6, 250e-6, 305e-6, 378e-6, 460e-6, 515e-6, 573e-6,
                 659e-6, 733e-6, 799e-6, 1199e-6, 860e-6, 933e-6, 993e-6,
                 1060e-6, 1125e-6], [5e-6], name='Lengthening', units='m')
# This data is the amount that the cable streched for each applied mass
# As the error on each of these measurements is the same, we will use a single
# value of error instead of another list containing the error for each point.

m50 = 0.05008
m1 = 0.10010
m2x = 0.20019
m2i = 0.20025
m5 = 0.50054
m5d = 0.50087

mass = u.Measured([0, m1, m2x, m2x+m1, m2x+m2i, m5, m1+m5, m2x+m5, m2x+m5+m1,
                   m5+m2x+m2i, m5+m5d+m2x+m2i+m1, m5+m5d, m5+m5d+m1,
                   m5+m5d+m2x, m5+m5d+m2x+m1, m5+m5d+m2x+m2i], [0.04],
                  name='Suspended Mass', units='m')

# This list is the combonation of weights that were used in each trial
# As the error on each of these measurements is the same, we will use a single
# value of error instead of another list containing the error for each point.

# Now that we have the data stored, we can plot the data, along with a line of
# best fit

plot = fit.Plot(dl, mass)  # This creates the plot and stores it as plot
plot.fit('linear')  # We can now say that we want to see a linear fit
plot.residuals()  # This tells the plot that we also want a residual plot
plot.show('file')
# Now that we have prepared the plot with everything, it is shown
