import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from numpy import percentile, random
from typing import Callable, List, Optional, cast
import numpy.typing as npt


Map = npt.NDArray[np.float32]
Transformer = Callable[[Map], Map]


class MapTransformer:
    @staticmethod
    def compose(*transformers: Transformer) -> Transformer:
        def composed_transformer(input_map: Map) -> Map:
            print("Starting map transformation pipeline...")
            result_map = input_map
            for i, transformer in enumerate(transformers, start=1):
                print(f"\t{i} of {len(transformers)} - {transformer.__name__}")
                result_map = transformer(result_map)
            print("Map transformation pipeline complete.")
            return result_map
        return composed_transformer

    @staticmethod
    def normalize() -> Transformer:
        def normalizer(input_map: Map) -> Map:
            min_val = input_map.min()
            max_val = input_map.max()
            if max_val - min_val == 0:
                return input_map
            return (input_map - min_val) / (max_val - min_val)
        return normalizer

    @staticmethod
    def threshold(min_threshold: float, max_threshold: float) -> Transformer:
        def thresholder(input_map: Map) -> Map:
            input_map[input_map < min_threshold] = min_threshold
            input_map[input_map > max_threshold] = max_threshold
            return input_map
        return thresholder

    @staticmethod
    def percentile_threshold(low_percentile: float, high_percentile: float) -> Transformer:
        def clipper(input_map: Map) -> Map:
            low, high = percentile(input_map[input_map > 0], [low_percentile, high_percentile])
            input_map[input_map < low] = low
            input_map[input_map > high] = high
            return input_map
        return clipper

    @staticmethod
    def power_transform(exponent: float) -> Transformer:
        def power_transformer(input_map: Map) -> Map:
            return input_map ** exponent
        return power_transformer

    @staticmethod
    def gaussian_blur(sigma: float) -> Transformer:
        from scipy.ndimage import gaussian_filter

        def blurrer(input_map: Map) -> Map:
            return gaussian_filter(input_map, sigma=sigma)
        return blurrer

    @staticmethod
    def sato_filter(sigmas: list[float]) -> Transformer:
        from skimage.filters import sato

        def sato_transformer(input_map: Map) -> Map:
            return sato(input_map, sigmas=cast(range, sigmas), black_ridges=False)
        return sato_transformer

    @staticmethod
    def scale(factor: float) -> Transformer:
        def scaler(input_map: Map) -> Map:
            return input_map * factor
        return scaler

    @staticmethod
    def invert() -> Transformer:
        def inverter(input_map: Map) -> Map:
            non_zero_mask = input_map != 0
            input_map[non_zero_mask] = 1.0 / input_map[non_zero_mask]
            return input_map
        return inverter

    @staticmethod
    def add_noise(amplitude: float) -> Transformer:
        def noiser(input_map: Map) -> Map:
            noise = random.uniform(0, amplitude, size=input_map.shape).astype(np.float32)
            return input_map + noise
        return noiser

    @staticmethod
    def save_distribution_plots(output_dir, prefix="Z_distribution"):
        def plotter(input_map: Map) -> Map:
            os.makedirs(output_dir, exist_ok=True)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # 1️⃣ Histogram (log freq)
            values = input_map[input_map > 0].flatten()
            axs[0].hist(values, bins=300, log=True, color='steelblue', alpha=0.8)
            axs[0].set_title('Histogram of Values (log freq)')
            axs[0].set_xlabel('Tile value')
            axs[0].set_ylabel('Frequency (log)')
            axs[0].grid(alpha=0.3)

            # 2️⃣ CDF (Cumulative Distribution)
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
            axs[1].plot(sorted_vals, cdf, color='steelblue')
            axs[1].set_title('Cumulative Distribution (CDF)')
            axs[1].set_xlabel('Tile value')
            axs[1].set_ylabel('Cumulative Percentage (%)')
            axs[1].grid(alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(
                output_dir,
                f"{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{prefix}.png"
            )
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"📊 Distribution plot saved to: {out_path}")
            return input_map
        return plotter


class MapTransformerBuilder:
    """Declarative builder for chaining map transformations, with optional distribution logging."""

    def __init__(self, output_dir: Optional[str] = None):
        self._transformers: List[Transformer] = []
        self._output_dir = output_dir

    def _add(self, transformer: Transformer):
        self._transformers.append(transformer)
        return self

    def normalize(self): return self._add(MapTransformer.normalize())
    def threshold(self, min_t, max_t): return self._add(MapTransformer.threshold(min_t, max_t))
    def percentile_threshold(self, low_p, high_p): return self._add(
        MapTransformer.percentile_threshold(low_p, high_p))

    def power_transform(self, exp): return self._add(MapTransformer.power_transform(exp))
    def gaussian_blur(self, sigma): return self._add(MapTransformer.gaussian_blur(sigma))
    def sato_filter(self, sigmas): return self._add(MapTransformer.sato_filter(sigmas))
    def scale(self, factor): return self._add(MapTransformer.scale(factor))
    def invert(self): return self._add(MapTransformer.invert())
    def add_noise(self, amplitude): return self._add(MapTransformer.add_noise(amplitude))

    def capture_distribution(self, prefix: str): return self._add(
        MapTransformer.save_distribution_plots(self._output_dir, prefix))

    def build(self) -> Transformer:
        """Returns the composed transformer pipeline."""
        return MapTransformer.compose(*self._transformers)
