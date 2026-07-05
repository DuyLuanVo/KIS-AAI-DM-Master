#!/usr/bin/env python3
"""
Script to upload local keyframe images to MinIO
"""
import os
import sys
from pathlib import Path
import boto3
from botocore.client import Config

# Add project root to path to load config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline import config as global_config


def upload_images():
    print("MinIO Keyframes Migration Utility")
    print("=================================")

    # MinIO details from config
    endpoint = global_config.MINIO_ENDPOINT
    access_key = global_config.MINIO_ACCESS_KEY
    secret_key = global_config.MINIO_SECRET_KEY
    bucket_name = global_config.MINIO_BUCKET
    secure = global_config.MINIO_SECURE

    # Local data root
    data_root = Path(global_config.DATA_ROOT)
    print(f"Data root: {data_root}")
    print(f"MinIO endpoint: {endpoint}")
    print(f"MinIO bucket: {bucket_name}")

    # Set up boto3 client
    s3_config = Config(
        signature_version="s3v4",
        s3={'addressing_style': 'path'}
    )
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}" if secure else f"http://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=s3_config
    )

    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' verified.")
    except Exception:
        print(f"Bucket '{bucket_name}' not found. Creating bucket...")
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to create bucket: {e}")
            return False

    # Scan for Keyframe folders
    # Usually data/Keyframes_L21/keyframes/video/001.jpg
    # Keyframes dir pattern is: Keyframes_*/keyframes/*/*.jpg
    keyframe_folders = list(data_root.glob("Keyframes_*/keyframes"))
    if not keyframe_folders:
        # Try without Keyframes_ prefix if none found, or list folders directly
        print("[WARN] No directories matching data/Keyframes_*/keyframes found. Scanning all subfolders in data root...")
        keyframe_folders = list(data_root.glob("**/keyframes"))

    if not keyframe_folders:
        print("[ERROR] No keyframes folder found.")
        return False

    uploaded_count = 0
    errors_count = 0

    for kf_dir in keyframe_folders:
        print(f"Scanning directory: {kf_dir}")
        # Subdirectories inside keyframes/ are video_ids
        video_dirs = [d for d in kf_dir.iterdir() if d.is_dir()]
        
        for video_dir in video_dirs:
            video_id = video_dir.name
            jpg_files = list(video_dir.glob("*.jpg"))
            print(f"  Video '{video_id}': found {len(jpg_files)} keyframes to upload.")
            
            for jpg_file in jpg_files:
                frame_name = jpg_file.name
                # Object key pattern: keyframes/{video_id}/{frame_name}
                object_key = f"keyframes/{video_id}/{frame_name}"
                
                try:
                    s3_client.upload_file(
                        Filename=str(jpg_file),
                        Bucket=bucket_name,
                        Key=object_key,
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )
                    uploaded_count += 1
                    if uploaded_count % 100 == 0:
                        print(f"    Uploaded {uploaded_count} images...")
                except Exception as e:
                    print(f"    [ERROR] Failed to upload {jpg_file.name}: {e}")
                    errors_count += 1

    print("=================================")
    print(f"Migration completed.")
    print(f"Total uploaded: {uploaded_count}")
    print(f"Total errors: {errors_count}")
    return errors_count == 0


if __name__ == "__main__":
    upload_images()
