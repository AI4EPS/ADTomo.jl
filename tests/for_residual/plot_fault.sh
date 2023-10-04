gmt begin 2 png
    gmt coast -R-124/-119.5/35.2/39.4 -JM15c -Ba -W0.5p
    gmt plot BayArea/fault/fault1.gmt -W0.3p,blue -Gwhite
    gmt plot BayArea/fault/fault3.gmt -W0.3p,green -Gwhite
gmt end