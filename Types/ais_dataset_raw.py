class AISDatasetRaw:

    def save(self, path: str):
        pass

    @staticmethod
    def load(path: str) -> "AISDatasetRaw":
        return AISDatasetRaw()
