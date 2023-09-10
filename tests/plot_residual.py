import pygmt
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

folder = "readin_data/1/"
alleve = pd.read_csv(folder+"alleve.csv")
allsta = pd.read_csv(folder+"allsta.csv")

hdf5_file = h5py.File(folder+'eve_ratio_p.h5','r')
dataset = hdf5_file['data']
eve_ratio_p = dataset[:]
hdf5_file = h5py.File(folder+'eve_ratio_s.h5','r')
dataset = hdf5_file['data']
eve_ratio_s = dataset[:]
hdf5_file = h5py.File(folder+'sta_ratio_p.h5','r')
dataset = hdf5_file['data']
sta_ratio_p = dataset[:]
hdf5_file = h5py.File(folder+'sta_ratio_s.h5','r')
dataset = hdf5_file['data']
sta_ratio_s = dataset[:]

fig = pygmt.Figure()
fig.basemap(region=[-123.5,-120.5,36,38.6], projection="M15c", frame=["a+f", "WSen"])
fig.grdimage(grid="@earth_relief_01m", interpolation="b", transparency=25)
fig.plot(x=allsta.lon, y=allsta.lat, style="c0.2c", color=sta_ratio_p, cmap="polar")
fig.colorbar(cmap="polar", position="JTR+jTR+o-1c/1c/0/1c/0.2c")
fig.colorbar()
fig.savefig(folder+"residual/mapsta_p.png")
fig = pygmt.Figure()
fig.basemap(region=[-123.5,-120.5,36,38.6], projection="M15c", frame=["a+f", "WSen"])
fig.grdimage(grid="@earth_relief_01m", interpolation="b", transparency=25)
fig.plot(x=allsta.lon, y=allsta.lat, style="c0.2c", color=sta_ratio_s, cmap="polar")
fig.colorbar(cmap="polar", position="JTR+jTR+o-1c/1c/0/1c/0.2c")
fig.colorbar()
fig.savefig(folder+"residual/mapsta_s.png")

numeve = len(alleve)
cmap = plt.get_cmap('seismic')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(numeve):
    ax.scatter(alleve.lon,alleve.lat,-alleve.z+4,marker='o',cmap='seismic',c=eve_ratio_p)
fig.savefig(folder+"residual/mapeve_p.png")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(numeve):
    ax.scatter(alleve.lon,alleve.lat,-alleve.z+4,marker='o',cmap='seismic',c=eve_ratio_s)
fig.savefig(folder+"residual/mapeve_s.png")