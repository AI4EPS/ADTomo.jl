import json
import pyproj
import pygmt
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt

with open("../local/demo/seismic_data/config.json", "r") as f:
    config = json.load(f)

proj = pyproj.Proj(
    f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
)

with open("../local/demo/readin_data/range.txt", "r") as rfile:
    m = int(rfile.readline())
    n = int(rfile.readline())
    l = int(rfile.readline())
    h = float(rfile.readline())
    dx = int(rfile.readline())
    dy = int(rfile.readline())
    dz = int(rfile.readline())

with open("../local/demo/readin_data/config.json","r") as f:
    config = json.load(f)["sta_eve"]

theta = math.radians(config["theta"])

folder = "../local/demo/readin_data/inv_P_0.005/"
ite = 100
vel = h5py.File(folder + f"post/post_{ite}.h5","r")["data"]
#vel = h5py.File("readin_data/store/new4/2/ratio/0.03_0.1/vratio_0.03_0.1.h5")["data"]

for i in range(16):
    with open(folder+"output/"+f"{i+1}.txt","w") as file:
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
                file.write(f"{lon} {lat} {vel[i,j,k]} 90 0.2 0.2\n")

''' 
x = (k-dx)*h
y = (j-dy)*h
nx = x*math.cos(theta) - y*math.sin(theta)
ny = y*math.cos(theta) + x*math.sin(theta)
lon, lat = proj(nx,ny,inverse=True)
print(lon,' ',lat)
'''
    
    
    