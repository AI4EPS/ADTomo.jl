#!/bin/bash

gmt begin mapsta_p png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -I+d -t25
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_p.txt
gmt end