from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm


def chunked_gaussian_blur(vx, vy, sigma=5, chunk_size=1024):
    overlap = int(3 * sigma)
    H, W = vx.shape

    blurred_vx = np.zeros_like(vx)
    blurred_vy = np.zeros_like(vy)

    num_y_chunks = (H + chunk_size - 1) // chunk_size
    num_x_chunks = (W + chunk_size - 1) // chunk_size
    total_chunks = num_y_chunks * num_x_chunks

    with tqdm(total=total_chunks, desc=f"Blurring field (σ={sigma}, chunk={chunk_size})") as pbar:
        for y0 in range(0, H, chunk_size):
            for x0 in range(0, W, chunk_size):
                y1 = min(y0 + chunk_size, H)
                x1 = min(x0 + chunk_size, W)

                # include overlap region
                y0_pad = max(y0 - overlap, 0)
                x0_pad = max(x0 - overlap, 0)
                y1_pad = min(y1 + overlap, H)
                x1_pad = min(x1 + overlap, W)

                sub_vx = vx[y0_pad:y1_pad, x0_pad:x1_pad]
                sub_vy = vy[y0_pad:y1_pad, x0_pad:x1_pad]

                sub_vx_blur = gaussian_filter(sub_vx, sigma=sigma)
                sub_vy_blur = gaussian_filter(sub_vy, sigma=sigma)

                # crop out overlap region when writing back
                y0_out = y0 - y0_pad
                x0_out = x0 - x0_pad
                y1_out = y1 - y0_pad
                x1_out = x1 - x0_pad

                blurred_vx[y0:y1, x0:x1] = sub_vx_blur[y0_out:y1_out, x0_out:x1_out]
                blurred_vy[y0:y1, x0:x1] = sub_vy_blur[y0_out:y1_out, x0_out:x1_out]

                pbar.update(1)

    return blurred_vx, blurred_vy
