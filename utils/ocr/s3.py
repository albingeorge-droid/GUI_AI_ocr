import os
from pathlib import Path
from typing import List, Tuple
import boto3

def list_pdfs_in_prefix(s3_client, bucket: str, prefix: str) -> List[str]:
    """
    Returns PDF keys under prefix (recursive).
    """
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.lower().endswith(".pdf") and not key.endswith("/"):
                keys.append(key)

    return sorted(keys)

def download_s3_key(s3_client, bucket: str, key: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, local_path)

def s3_key_to_local_path(base_dir: str, prefix: str, key: str) -> str:
    """
    Keep folder structure under prefix.
    """
    rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
    return str(Path(base_dir) / rel)
