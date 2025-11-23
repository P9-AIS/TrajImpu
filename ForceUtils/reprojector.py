from osgeo import gdal, osr
import numpy as np
import sys
from pyproj import Transformer


from ForceTypes.area import Area
from ForceTypes.espg3034_coord import Espg3034Coord


from rasterio.warp import transform_bounds


class Reprojector:

    @staticmethod
    def reproject_png_3034_to_3857(image_path: str, output_path: str, area: Area):
        print(f"Reading {image_path}...")
        src_ds = gdal.Open(image_path)
        if src_ds is None:
            raise RuntimeError(f"Failed to open {image_path}")

        # Create in-memory dataset with georeferencing
        print("Applying georeferencing (EPSG:3034)...")
        mem_driver = gdal.GetDriverByName("MEM")
        mem_ds = mem_driver.CreateCopy("", src_ds)

        # Set geotransform: upper-left corner coords and pixel size
        # Format: [x_min, pixel_width, 0, y_max, 0, -pixel_height]
        ulx, uly = area.bottom_left.E, area.top_right.N
        lrx, lry = area.top_right.E, area.bottom_left.N

        width = mem_ds.RasterXSize
        height = mem_ds.RasterYSize

        pixel_width = (lrx - ulx) / width
        pixel_height = (uly - lry) / height

        geotransform = [ulx, pixel_width, 0, uly, 0, -pixel_height]
        mem_ds.SetGeoTransform(geotransform)

        # Set projection (EPSG:3034)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3034)
        mem_ds.SetProjection(srs.ExportToWkt())

        # Step 2: Warp to EPSG:3857 in memory
        print("Reprojecting to EPSG:3857...")

        # Target bounds in EPSG:3857
        transformer = Transformer.from_crs("EPSG:3034", "EPSG:3857", always_xy=True)
        te = transformer.transform_bounds(ulx, lry, lrx, uly)

        warp_options = gdal.WarpOptions(
            srcSRS="EPSG:3034",
            dstSRS="EPSG:3857",
            resampleAlg="bilinear",
            outputBounds=te,
            dstAlpha=True,
            format="MEM",
        )

        warped_ds = gdal.Warp("", mem_ds, options=warp_options)
        if warped_ds is None:
            raise RuntimeError("Warping failed")

        # Get the geotransform of the warped image
        gt = warped_ds.GetGeoTransform()
        warped_width = warped_ds.RasterXSize
        warped_height = warped_ds.RasterYSize

        # Calculate coordinates
        # Top-left pixel (0, 0)
        top_left_x = gt[0]
        top_left_y = gt[3]

        # Bottom-left pixel (0, height)
        bottom_left_x = gt[0]
        bottom_left_y = gt[3] + gt[5] * warped_height

        # Top-right pixel (width, 0)
        top_right_x = gt[0] + gt[1] * warped_width
        top_right_y = gt[3]

        # Bottom-right pixel (width, height)
        bottom_right_x = gt[0] + gt[1] * warped_width
        bottom_right_y = gt[3] + gt[5] * warped_height

        print(f"\nOutput image coordinates (EPSG:3857):")
        print(f"Bottom-left: ({bottom_left_x:.2f}, {bottom_left_y:.2f})")
        print(f"Top-right: ({top_right_x:.2f}, {top_right_y:.2f})")
        print(f"Top-left: ({top_left_x:.2f}, {top_left_y:.2f})")
        print(f"Bottom-right: ({bottom_right_x:.2f}, {bottom_right_y:.2f})")

        # Step 3: Export to PNG (RGB + Alpha if present)
        print(f"\nWriting {output_path}...")

        # Check how many bands we have (should be 4 with alpha from dstAlpha)
        num_bands = warped_ds.RasterCount

        # Create in-memory dataset with RGBA bands
        mem_rgba = mem_driver.Create(
            "", warped_ds.RasterXSize, warped_ds.RasterYSize, num_bands, gdal.GDT_Byte
        )

        # Copy all bands including alpha
        for i in range(1, num_bands + 1):
            band = warped_ds.GetRasterBand(i)
            data = band.ReadAsArray()
            mem_rgba.GetRasterBand(i).WriteArray(data)

        # Use CreateCopy to write PNG
        png_driver = gdal.GetDriverByName("PNG")
        out_ds = png_driver.CreateCopy(output_path, mem_rgba)

        # Cleanup
        out_ds = None
        mem_rgba = None
        warped_ds = None
        mem_ds = None
        src_ds = None

        print("Reprojection and export completed.")
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:3034", always_xy=True)
        bottom_left_x, bottom_left_y = transformer.transform(te[0], te[1])
        top_right_x, top_right_y = transformer.transform(te[2], te[3])

        new_area = Area(
            bottom_left=Espg3034Coord(E=bottom_left_x, N=bottom_left_y),
            top_right=Espg3034Coord(E=top_right_x, N=top_right_y)
        )

        return new_area
