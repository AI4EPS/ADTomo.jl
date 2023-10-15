#!/bin/bash

cd ../local/demo/readin_data/
gmt begin range png
    gmt basemap -R-124/-119.5/35.2/39.4 -JM15c -Ba -t50
    gmt grdimage @earth_relief_01m -Baf -BWSen
    gmt colorbar

    gmt plot -W2p,royalblue -L << EOF
    -123.80712028585117 38.3731377818302
    -122.23763624860433 39.159851389716955
    -119.99992529287142 36.255171597377505
    -121.53301300000949 35.49760222861597
EOF
    #gmt plot BayArea/range_2000.dat -W2p,yellow -L
    #gmt plot BayArea/range_2007.dat -W2p,skyblue -L
    awk -F ',' 'NR > 1 {print $4 " " $5}' sta_eve/alleve.csv > sta_eve/eve.txt
    gmt plot -Sc0.2c -Gred sta_eve/eve.txt
    awk -F ',' 'NR > 1 {print $4 " " $5}' sta_eve/allsta.csv > sta_eve/sta.txt
    gmt plot -St0.3c -Gblue sta_eve/sta.txt
gmt end