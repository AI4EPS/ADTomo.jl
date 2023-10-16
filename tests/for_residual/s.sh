#!/bin/bash

gmt begin mapsta_s png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -t25
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_s.txt
gmt end

gmt begin mapeve_s_0 png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -t25
    gmt colorbar
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_S.txt
gmt end

gmt begin mapeve_s_1 png
    gmt basemap -R-123.5/-120.5/-30/4 -JX8c/5c -Ba
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_s.txt
gmt end

gmt begin mapeve_s_2 png
    gmt basemap -R36/38.6/-30/4 -JX8c/5c -Ba
    gmt makecpt -Cpolar
    gmt plot -Sc0.2c -C for_s.txt
gmt end