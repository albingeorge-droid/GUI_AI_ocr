"""
FAST S3 folder audit (parallel) with retries + backoff

Counts:
- total parent folders under PREFIX
- how many have ocr/
- how many have fext/
Prints:
- list of folders missing ocr/
- list of folders missing fext/

✅ AWS creds assumed in env
✅ Safe parallelism + exponential backoff on throttling/transient errors
"""

import boto3
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError

BUCKET = "guai"
PREFIX = "Haryana1/"   # must end with '/'

# Tune these
MAX_WORKERS = 20          # try 10-30
MAX_RETRIES = 6           # retry attempts per request
BASE_BACKOFF = 0.4        # seconds

# If you know your bucket region, set it for fewer redirects/faster calls:
# s3 = boto3.client("s3", region_name="ap-south-1")
s3 = boto3.client("s3")


def is_retryable_error(e: Exception) -> bool:
    if isinstance(e, (EndpointConnectionError, ConnectionClosedError)):
        return True
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        # common retryable S3 errors
        return code in {
            "SlowDown",
            "Throttling",
            "ThrottlingException",
            "RequestTimeout",
            "RequestTimeoutException",
            "InternalError",
            "ServiceUnavailable",
            "500",
            "503",
        }
    return False


def s3_list_objects_v2_with_retries(**kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return s3.list_objects_v2(**kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES or not is_retryable_error(e):
                raise
            # exponential backoff + jitter
            sleep_s = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)


def list_parent_folders(bucket: str, prefix: str):
    """List all immediate subfolders under PREFIX using pagination."""
    parents = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3_list_objects_v2_with_retries(**kwargs)
        parents.extend([cp["Prefix"] for cp in resp.get("CommonPrefixes", [])])

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return parents


def check_one_parent(parent_prefix: str):
    """
    For a given parent folder (e.g., Haryana/1_CLU_GN-1199/),
    check if it contains subfolders ocr/ and fext/
    """
    resp = s3_list_objects_v2_with_retries(
        Bucket=BUCKET,
        Prefix=parent_prefix,
        Delimiter="/",
        MaxKeys=1000
    )
    subs = {cp["Prefix"] for cp in resp.get("CommonPrefixes", [])}

    has_ocr = (parent_prefix + "ocr/") in subs
    has_fext = (parent_prefix + "fext/") in subs
    return parent_prefix, has_ocr, has_fext


def main():
    parents = list_parent_folders(BUCKET, PREFIX)
    total = len(parents)

    print(f"Found {total} folders under s3://{BUCKET}/{PREFIX}")
    print(f"Checking subfolders with {MAX_WORKERS} workers...\n")

    missing_ocr = []
    missing_fext = []
    ocr_count = 0
    fext_count = 0

    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(check_one_parent, p) for p in parents]

        for fut in as_completed(futures):
            done += 1
            try:
                parent, has_ocr, has_fext = fut.result()
            except Exception as e:
                # treat failures as missing + report
                parent = "UNKNOWN_PARENT"
                has_ocr = False
                has_fext = False
                print(f"[ERROR] A folder check failed: {e}")

            if has_ocr:
                ocr_count += 1
            else:
                missing_ocr.append(parent)

            if has_fext:
                fext_count += 1
            else:
                missing_fext.append(parent)

            # progress every 100
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"Progress: {done}/{total} | {rate:.1f} folders/s | ETA ~ {eta:.0f}s")

    print("\n========== S3 FOLDER AUDIT ==========")
    print(f"Bucket              : {BUCKET}")
    print(f"Prefix              : {PREFIX}")
    print("-------------------------------------")
    print(f"Total folders       : {total}")
    print(f"Folders with ocr/   : {ocr_count}")
    print(f"Folders with fext/  : {fext_count}")
    print("-------------------------------------")
    print(f"Missing ocr/ count  : {len(missing_ocr)}")
    print(f"Missing fext/ count : {len(missing_fext)}")
    print("=====================================\n")

    # Sort for nicer output
    missing_ocr_sorted = sorted([p for p in missing_ocr if p != "UNKNOWN_PARENT"])
    missing_fext_sorted = sorted([p for p in missing_fext if p != "UNKNOWN_PARENT"])

    if missing_ocr_sorted:
        print("Folders missing ocr/:")
        for p in missing_ocr_sorted:
            print(" -", p)
        print()

    if missing_fext_sorted:
        print("Folders missing fext/:")
        for p in missing_fext_sorted:
            print(" -", p)
        print()


if __name__ == "__main__":
    main()
