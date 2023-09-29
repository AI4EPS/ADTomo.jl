#!/bin/bash

gmt begin range png
    gmt basemap -R-124/-119.5/35.2/39.4 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -I+d -t25
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot range.dat -W2p,yellow -L
    gmt plot -Sc0.1c -Gred -t50 events.txt
    gmt plot -St0.1c -Gblue -t50 stations.txt
gmt end