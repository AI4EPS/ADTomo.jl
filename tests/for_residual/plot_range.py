import pygmt
import numpy as np

fig = pygmt.Figure()

fig.basemap(region=[-124, -119.5, 35.2, 39.4], projection="M15c", frame=True)
fig.grdimage(grid="@earth_relief_01m", transparency=50, shading=True, frame=["WSen"])
fig.colorbar()
#pygmt.makecpt(cmap="polar")

data = np.loadtxt("BayArea/range/range_0.dat")
fig.plot(x=data[:,0], y=data[:,1], pen="2p,royalblue")

# fig.plot(data="BayArea/range/range_2000.dat", style="L2p,yellow", is_line=True)
# fig.plot(data="BayArea/range/range_2007.dat", style="L2p,skyblue", is_line=True)

data = np.loadtxt("BayArea/range/events.txt")
fig.plot(x=data[:,0],y=data[:,1],style="c0.2c", fill="red",transparency= 80,label="events+S0.25c")
data = np.loadtxt("BayArea/range/stations.txt")
fig.plot(x=data[:,0],y=data[:,1],style="t0.2c", fill="blue",transparency= 80,label="stations+S0.25c")

fig.legend(transparency=30)
fig.savefig("range.png")