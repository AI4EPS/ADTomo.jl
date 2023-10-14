using JSON

region = "BayArea/"
config = Dict(
    "sta_eve" => Dict(
        "region" => region,
        "h" => 2.0,
        "theta" => 32,
        "p_requirement" => 0.8,
        "s_requirement" => 0.6,
        "l_x" => 45,
        "s_x" => -35,
        "l_y" => 1e6,
        "s_y" => -1e6,
        "l_z" => 15,
        "s_z" => -1e6,
        "eve_picks" => 10,
        "sta_picks" => 10,
        "eve_eps" => 2.2,
        "sta_eps" => 1.414,
        "eve_ratio" => 25
    ),
    "gene_obs" => Dict(
        "prange" => 1.5,
        "srange" => 3,
        "p_requirement" => 0.8,
        "s_requirement" => 0.6,
        "p_times" => 1.042,
        "s_times" => 1.083,
        "bins_p" => 6000,
        "bins_s" => 5000
    ),
)

json_str = JSON.json(config, 3)  
json_str = json_str * "\n"

folder = "../local/"*region*"readin_data/"
if !isdir(folder) mkdir(folder) end 
json_file = folder * "config.json"

open(json_file, "w") do io
    write(io, json_str)
end



region = "demo/"
config = Dict(
    "sta_eve" => Dict(
        "region" => region,
        "h" => 1.0,
        "theta" => 0,
        "p_requirement" => 0.8,
        "s_requirement" => 0.6,
        "l_x" => 1e6,
        "s_x" => -1e6,
        "l_y" => 1e6,
        "s_y" => -1e6,
        "l_z" => 15,
        "s_z" => -1e6,
        "eve_picks" => 10,
        "sta_picks" => 10,
        "eve_eps" => 1.732,
        "sta_eps" => 1.414,
        "eve_ratio" => 25
    ),
    "gene_obs" => Dict(
        "prange" => 1.5,
        "srange" => 3,
        "p_requirement" => 0.8,
        "s_requirement" => 0.6,
        "p_times" => 1.027,
        "s_times" => 1.064,
        "bins_p" => 1000,
        "bins_s" => 1000
    ),
)

json_str = JSON.json(config, 3)  
json_str = json_str * "\n"

folder = "../local/"*region*"readin_data/"
if !isdir(folder) mkdir(folder) end 
json_file = folder * "config.json"

open(json_file, "w") do io
    write(io, json_str)
end