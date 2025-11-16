import numpy as np
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import os

from ForceTypes.area import Area
from ForceTypes.espg3034_coord import Espg3034Coord


class Reprojector:

    @staticmethod
    def reproject_png_3034_to_3857(image_path: str, output_path: str, area: Area) -> tuple[str, Area]:
        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img)
        height, width = arr.shape[:2]

        src_crs = "EPSG:3034"
        dst_crs = "EPSG:3857"

        src_transform = from_bounds(area.bottom_left.E, area.bottom_left.N,
                                    area.top_right.E, area.top_right.N, width, height)

        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, width, height,
            area.bottom_left.E, area.bottom_left.N, area.top_right.E, area.top_right.N
        )

        assert dst_h is not None and dst_w is not None, f"Invalid dimensions: {dst_w} x {dst_h}"

        dst_arr = np.zeros((4, dst_h, dst_w), dtype=np.uint8)

        for band in range(4):
            reproject(
                source=arr[:, :, band],
                destination=dst_arr[band],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

        out_arr = np.moveaxis(dst_arr, 0, -1)
        out_img = Image.fromarray(out_arr, "RGBA")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_img.save(output_path)

        bbox_3857 = (
            dst_transform.c,  # left
            dst_transform.f + dst_h * dst_transform.e,  # bottom
            dst_transform.c + dst_w * dst_transform.a,  # right
            dst_transform.f  # top
        )

        bbox_3034 = transform_bounds(dst_crs, src_crs, *bbox_3857)

        area_3034 = Area(
            bottom_left=Espg3034Coord(E=bbox_3034[0], N=bbox_3034[1]),
            top_right=Espg3034Coord(E=bbox_3034[2], N=bbox_3034[3])
        )

        return output_path, area_3034
