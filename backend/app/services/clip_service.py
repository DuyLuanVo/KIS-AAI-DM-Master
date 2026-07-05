"""
CLIP service for text and image embedding
"""
import base64
import io
from typing import List

import clip
import torch
from loguru import logger
from PIL import Image


class CLIPService:
    """Service for CLIP text and image embedding"""

    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize CLIP model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {model_name} on {self.device}")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        logger.info("CLIP model loaded successfully")

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """
        Encode list of texts to embeddings

        Args:
            texts: List of text queries

        Returns:
            List of embedding vectors (512-dim)
        """
        try:
            # Tokenize texts
            text_tokens = clip.tokenize(texts).to(self.device)

            # Get embeddings
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to list of lists
            embeddings = text_features.cpu().numpy().tolist()

            logger.info(f"Encoded {len(texts)} texts to embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise

    def encode_image_from_base64(self, image_base64: str) -> List[float]:
        """
        Encode base64 image to embedding

        Args:
            image_base64: Base64 encoded image string

        Returns:
            Embedding vector (512-dim)
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            return self.encode_image_from_pil(image)

        except Exception as e:
            logger.error(f"Base64 image encoding failed: {e}")
            raise

    def encode_image_from_pil(self, image: Image.Image) -> List[float]:
        """
        Encode PIL image to embedding

        Args:
            image: PIL Image object

        Returns:
            Embedding vector (512-dim)
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Get embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to list
            embedding = image_features.cpu().numpy().squeeze().tolist()

            logger.info("Encoded image to embedding")
            return embedding

        except Exception as e:
            logger.error(f"PIL image encoding failed: {e}")
            raise

    def encode_image_from_path(self, image_path: str) -> List[float]:
        """
        Encode image from file path to embedding

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector (512-dim)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.encode_image_from_pil(image)

        except Exception as e:
            logger.error(f"Image path encoding failed: {e}")
            raise


# Global CLIP service instance
clip_service = CLIPService()
