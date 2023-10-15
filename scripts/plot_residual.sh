#!/bin/bash

# this part generate the data for plot
cd ../local/demo/readin_data/
awk -F ',' 'NR > 1 {print $4 " " $5}' sta_eve/allsta.csv |
paste -d ' ' - for_P/residual/sta_ratio_p.txt > for_P/residual/plot_sta_ratio_p.txt

# this part use GMT to plot useful figures
gmt begin ratio_p png
    gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
    gmt grdimage @earth_relief_01m -Baf -BWSen -t25
    gmt colorbar
    gmt makecpt -Cpolar -T-1/1
    gmt plot -Sc0.2c -C for_P/residual/plot_sta_ratio_p.txt
gmt end