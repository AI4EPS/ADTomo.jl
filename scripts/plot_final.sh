cd ../local/demo/

gmt begin readin_data/inv_P_0.005/output/9 png
    gmt makecpt -Cpolar -I -T5.08/7.08/0.035
    gmt basemap -R-123.2/-120/35.8/39 -JM15c -Ba
    gmt plot -Sj -C readin_data/inv_P_0.005/output/9.txt
    gmt coast -R-123.2/-120/35.8/39 -JM15c -Ba -W0.8p

    gmt plot seismic_data/SHP/ca_offshore.shp -W0.3p,lightblue4
    gmt plot seismic_data/SHP/fault_areas.shp -W0.3p,lightblue4
    gmt plot seismic_data/SHP/Qfaults_US_Database.shp -W0.3p,lightblue4

    gmt text -F+f30p<< EOF
    -121.5 38.8 6 km
EOF
    gmt colorbar -DjMR+w6i/0.3i+v -C -Bxa0.1f0.1 -By+l"vel" 
gmt end