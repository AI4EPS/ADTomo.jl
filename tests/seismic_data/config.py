# %%
config = {
    "degree2km": 111.1949,
    "provider": None,
    "network": None,
    "station": None,
    "channel": "HH*,BH*,EH*,HN*",
    "channel_priorities": (
        "HH[ZNE12]",
        "BH[ZNE12]",
        "MH[ZNE12]",
        "EH[ZNE12]",
        "LH[ZNE12]",
        "HL[ZNE12]",
        "BL[ZNE12]",
        "ML[ZNE12]",
        "EL[ZNE12]",
        "LL[ZNE12]",
        "SH[ZNE12]",
    ),
    "location_priorities": (
        "",
        "00",
        "10",
        "01",
        "20",
        "02",
        "30",
        "03",
        "40",
        "04",
        "50",
        "05",
        "60",
        "06",
        "70",
        "07",
        "80",
        "08",
        "90",
        "09",
    ),
    "phasenet": {},
    "gamma": {},
    "cctorch": {},
    "adloc": {},
    "hypodd": {},
    "growclust": {},
}

# %%
config_region = {}

region = "BayArea"
config_region[region] = {
    "region": region,
    "starttime": "2022-01-01T00:00:00",
    "endtime": "2023-01-01T00:00:00",
    "minlatitude": 37.42 - 1.0,
    "maxlatitude": 37.42 + 1.0,
    "minlongitude": -121.97 - 1.0,
    "maxlongitude": -121.97 + 1.0,
    "provider": ["NCEDC"],
    "degree2km": 111.19492474777779,
    "channel_priorities": (
        "HH[ZNE12]",
        "BH[ZNE12]",
    ),
}


# %%
