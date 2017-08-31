import qexpy as q
q.plot_engine = 'mpl'
fig1 = q.MakePlot(xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ydata = [0.9, 1.4, 2.5, 4.2, 5.7, 6., 7.3, 7.1, 8.9, 10.8], yerr = 0.5, xname = 'length', xunits='m', yname = 'force', yunits='N', data_name = 'mydata')
fig1.fit('linear')
fig1.add_residuals()
fig1.show()