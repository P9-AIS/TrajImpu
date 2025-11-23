original to .tif
`gdal_translate -a_ullr 3592900.0 3475350.0 4399500.0 3055500.0 -a_srs EPSG:3034 original.png original.tif`

original .tif to 3857 .tif (correct cut)
```
gdalwarp \
  -s_srs EPSG:3034 \
  -t_srs EPSG:3857 \
  -r bilinear \
  -te 394242.7653415356 7210032.3453380335 1888495.0767057044 7985639.975621158 \
  -dstalpha \
  original.tif original_3857_correct_cut.tif
```


`.tif` to `.png`
`gdal_translate -of PNG -b 1 -b 2 -b 3 ./yourimage_3857.tif leaflet_ready_full.png`


---

original .tif to 3857 full tif (wrong cut)
`gdalwarp -t_srs EPSG:3857 original.tif original_3857_wrong_cut.tif`


