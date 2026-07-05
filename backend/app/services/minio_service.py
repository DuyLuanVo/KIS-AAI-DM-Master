"""
MinIO service for object storage operations and generating pre-signed URLs
"""
import boto3
from botocore.client import Config
from app.config import settings
from loguru import logger


class MinIOService:
    """Service for interacting with MinIO Object Storage"""

    def __init__(self):
        """Initialize MinIO (S3) client"""
        logger.info(f"Initializing MinIO client targeting {settings.minio_endpoint}")
        
        # Configure connection
        # Since we run locally, we set signature_version to s3v4 and addressing_style to path
        # to ensure compatibility with MinIO endpoints
        s3_config = Config(
            signature_version="s3v4",
            s3={'addressing_style': 'path'}
        )
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.minio_endpoint}" if settings.minio_secure else f"http://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=s3_config
        )
        self.bucket_name = settings.minio_bucket

    def generate_presigned_url(self, object_key: str, expires_in: int = 3600) -> str:
        """
        Generate a pre-signed URL for downloading an object from MinIO
        
        Args:
            object_key: The relative path/key of the object on MinIO
            expires_in: Expiration time in seconds (default: 1 hour)
            
        Returns:
            Pre-signed URL string
        """
        try:
            # Clean leading slash if present
            clean_key = object_key.lstrip("/")
            
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": clean_key
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate pre-signed URL for key {object_key}: {e}")
            raise


# Global MinIO service instance
minio_service = MinIOService()
