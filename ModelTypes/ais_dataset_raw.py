import os
import numpy as np
import pickle


class AISDatasetRaw:
    _dataset: np.ndarray

    def __init__(self, data: np.ndarray):
        if data is not None:
            self._dataset = data
        else:
            raise ValueError("Data must be provided to initialize AisDatasetRaw")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving raw ais dataset to '{path}'")
        np.savez_compressed(
            path,
            raw_ais_dataset_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            dataset=self._dataset
        )
        print(f"Saved raw ais dataset of size {self._dataset.size}\n")

    @staticmethod
    def load(path: str) -> "AISDatasetRaw":
        print(f"Loading raw ais dataset from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            dataset: AISDatasetRaw = pickle.loads(data['raw_ais_dataset_object'].item())
            dataset._dataset = data['dataset']
        print(f"Loaded raw ais dataset of size {dataset._dataset.size}\n")
        return dataset

    # method to combine two AisDatasetRaw objects
    def combine(self, other: "AISDatasetRaw") -> "AISDatasetRaw":
        combined_data = np.vstack((self._dataset, other._dataset))
        return AISDatasetRaw(combined_data)
