import qexpy as q
q.plot_engine='mpl'

fig = q.MakePlot(xdata=[1,2,3,4,5],ydata=[2,4,5,8,11], yerr=1)
fig.fit("pol2")
fig.show()