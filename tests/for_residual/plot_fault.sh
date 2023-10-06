gmt begin BayArea/velocity/vel_ratio/1 png
    gmt makecpt -Cpolar -I -T1.4/2.1/0.03
    gmt basemap -R-124/-119.5/35.2/39.4 -JM15c -Ba
    gmt plot -Sj -C ../readin_data/store/new4/2/ratio/0.03_0.1/output/1.txt
    gmt coast -R-124/-119.5/35.2/39.4 -JM15c -Ba -W0.5p
    #gmt plot BayArea/range/range_0.dat -W2p,black -L
    gmt plot BayArea/fault/fault1.gmt -W0.3p,lightblue4
    gmt plot BayArea/fault/fault3.gmt -W0.3p,lightblue4
    gmt text -F+f20p<< EOF
    -121.5 39.2 -2 km
EOF
    gmt colorbar -DjMR+w4i/0.25i+v -C -Bxa0.1f0.1 -By+l"ratio" 
gmt end