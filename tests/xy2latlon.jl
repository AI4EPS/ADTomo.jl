using JSON
using PyCall

# Load pyproj using PyCall
pyproj = pyimport("pyproj")

# Load the configuration file
config = JSON.parsefile("config.json")

# Define the proj
proj = pyproj.Proj(
    "+proj=sterea +lon_0=$( (config["minlongitude"] + config["maxlongitude"]) / 2 ) +lat_0=$( (config["minlatitude"] + config["maxlatitude"]) / 2 ) +units=km"
)

# Convert lon, lat to x, y
lon, lat = [-121, -120], [37, 36]
x, y = proj(lon, lat)
println(x, " ", y)

# Convert x, y back to lon, lat
lon, lat = proj(x, y, inverse=true)
println(lon, " ", lat)