#!/bin/bash

gmt begin median_p png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -t25
    gmt colorbar
    gmt makecpt -Cpolar -T-0.5/0.5
    gmt plot -Sc0.2c -C for_p.txt
gmt end

gmt begin mapeve_p_0 png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -t25
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_P.txt
gmt end

gmt begin mapeve_p_1 png
    gmt basemap -R-123.5/-120.5/-30/4 -JX8c/5c -Ba
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_P.txt
gmt end

gmt begin mapeve_p_2 png
    gmt basemap -R36/38.6/-30/4 -JX8c/5c -Ba
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_P.txt
gmt end
