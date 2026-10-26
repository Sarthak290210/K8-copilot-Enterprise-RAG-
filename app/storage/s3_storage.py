"""Storage in an S3 bucket (S3_CACHE_BUCKET, AWS_REGION; credentials from the usual AWS
environment/profile chain)."""

from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.storage.storage_backend import StorageBackend


class S3Storage(StorageBackend):
    def __init__(self, bucket: str | None = None, region: str | None = None) -> None:
        self.bucket = bucket or settings.s3_cache_bucket
        self.client: Any = boto3.client("s3", region_name=region or settings.aws_region)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def save_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read_bytes(self, key: str) -> bytes | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return response["Body"].read()

    def url_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"
