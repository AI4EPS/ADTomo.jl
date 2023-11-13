# %%
import json
import pyproj


# %%
with open("config.json", "r") as f:
    config = json.load(f)

# %%
proj = pyproj.Proj(
    f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
)

# %%
lon, lat = [-121, -120], [37, 36]
x, y = proj(lon, lat)
print(x, y)

# %%
lon, lat = proj(x, y, inverse=True)
print(lon, lat)
# %%
