#!/bin/bash

gmt begin range png,pdf
    gmt basemap -R-124/-119.5/35.2/39.4 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -t100 -BWSen 
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot BayArea/range/range_0.dat -W2p,royalblue -L
    #gmt plot BayArea/range_2000.dat -W2p,yellow -L
    #gmt plot BayArea/range_2007.dat -W2p,skyblue -L
    gmt plot -Sc0.2c -Gred BayArea/range/events.txt
    gmt plot -St0.3c -Gblue@50 BayArea/range/stations.txt
gmt end