from abc import ABC, abstractmethod
import datetime as dt
from ForceTypes.area import Area
from PIL import Image


class IForceDataUploadHandler(ABC):
    @abstractmethod
    def upload_image(self, image: Image.Image, name: str, area: Area) -> None:
        pass
