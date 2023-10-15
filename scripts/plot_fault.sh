gmt begin BayArea/velocity/final/ratio_9 png
    gmt makecpt -Cpolar -I -T1.4/2.09/0.035
    gmt basemap -R-123.2/-120/35.8/39 -JM15c -Ba
    gmt plot -Sj -C ../readin_data/store/new4/2/ratio/0.03_0.1/output/9.txt
    gmt coast -R-123.2/-120/35.8/39 -JM15c -Ba -W0.8p
    #gmt plot BayArea/range/range_0.dat -W2p,black -L
    gmt plot BayArea/fault/fault1.gmt -W0.3p,lightblue4
    gmt plot BayArea/fault/fault3.gmt -W0.3p,lightblue4
    gmt text -F+f30p<< EOF
    -121.5 38.8 14 km
EOF
    gmt colorbar -DjMR+w6i/0.3i+v -C -Bxa0.1f0.1 -By+l"ratio" 
gmt end