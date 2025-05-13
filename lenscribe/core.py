from typing import Protocol
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path.
    """
    image = Image.open(image_path).convert("RGB")
    return image

class ImageProcessor(Protocol):
    def __init__(self, question: str, image_path: str) -> None:
        pass

    def configure(self) -> None:
        """
        Configure the image processor and model.
        """
        pass

    def process(self, **kwargs) -> Image.Image:
        """
        Process an image and return the processed image.
        """
        pass


class BlipImageProcessor:
    def __init__(self, question: str, image_path: str) -> None:
        """
        Initialize the BlipImageProcessor with a question and image path.
        """
        self.question = question
        self.image_path = image_path
        self.processor = None
        self.model = None

    def configure(self) -> None:
        """
        Configure the Blip image processor with the given settings.
        """
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    def process(self, **kwargs) -> Image.Image:
        """
        Process an image and return the processed image.
        """
        image = load_image(self.image_path)
        inputs = self.processor(image, self.question, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        print(f"Question: {self.question}")
        print(f"Answer: {answer}")
        return answer