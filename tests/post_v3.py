import json
import pyproj
import pygmt
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt

with open("seismic_data/BayArea/config.json", "r") as f:
    config = json.load(f)

proj = pyproj.Proj(
    f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
)

with open("readin_data/range.txt", "r") as rfile:
    m = int(rfile.readline())
    n = int(rfile.readline())
    l = int(rfile.readline())
    h = float(rfile.readline())
    dx = int(rfile.readline())
    dy = int(rfile.readline())
    dz = int(rfile.readline())

theta = math.radians(32)

folder = "readin_data/store/new4/2/inv_S_0.1/intermediate/"
ite = 101
#vel = h5py.File(folder + f"post_{ite}.h5","r")["data"]
#check_1 = h5py.File("readin_data/velocity/vel_2/vel_check_s_20.h5")["data"]
#check_2 = h5py.File("readin_data/store/new4/2/check_S_0.1_20/intermediate/post_100.h5")["data"]
vel = h5py.File("readin_data/store/new4/2/ratio/0.03_0.1/vratio_0.03_0.1.h5")["data"]
#folder = "readin_data/store/new4/2/inv_S_0.1/output/"
folder = "readin_data/store/new4/2/ratio/0.03_0.1/output/"
for i in range(16):
    with open(folder+f"{i+1}.txt","w") as file:
        print(vel[i,0,0])
        for j in range(n):
            for k in range(m):
                #if np.abs(check_1[i,j,k]-check_2[i,j,k]) > 0.2:
                    #print(check_1[i,j,k],' ',check_2[i,j,k])
                #    continue
                    
                y = (j+1-dy)*h
                x = (k+1-dx)*h
                
                nx = x*math.cos(theta) - y*math.sin(theta)
                ny = y*math.cos(theta) + x*math.sin(theta)
                lon, lat = proj(nx,ny,inverse=True)
                file.write(f"{lon} {lat} {vel[i,j,k]} 122 0.2 0.2\n")

''' 
x = (k-dx)*h
y = (j-dy)*h
nx = x*math.cos(theta) - y*math.sin(theta)
ny = y*math.cos(theta) + x*math.sin(theta)
lon, lat = proj(nx,ny,inverse=True)
print(lon,' ',lat)
'''
    
    
    