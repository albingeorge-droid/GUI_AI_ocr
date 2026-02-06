#!/usr/bin/env python

# uv run uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# http://127.0.0.1:8000/docs

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_pipeline.stage_01_ocr import run_stage_01_ocr_from_s3
from data_pipeline.stage_02_fext import run_stage_02_fext_from_s3
from data_pipeline.stage_03_data_cleaning import run_stage_03_clean_from_s3


app = FastAPI(title="GUI_AI_OCR Backend", version="1.0.0")

# Keep it small; these are heavy jobs (Bedrock, S3 IO).
_EXECUTOR = ThreadPoolExecutor(max_workers=2)

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _submit_job(fn: Callable[..., None], *, job_name: str, kwargs: Dict[str, Any]) -> str:
    job_id = uuid4().hex

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "status": "queued",  # queued | running | success | error
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "traceback": None,
            "kwargs": kwargs,
        }

    def _runner():
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "running"
            _JOBS[job_id]["started_at"] = _utc_now()

        try:
            fn(**kwargs)

            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "success"
                _JOBS[job_id]["finished_at"] = _utc_now()

        except Exception as e:
            tb = traceback.format_exc()
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["finished_at"] = _utc_now()
                _JOBS[job_id]["error"] = str(e)
                _JOBS[job_id]["traceback"] = tb

    _EXECUTOR.submit(_runner)
    return job_id


# -------------------------
# Request models
# -------------------------
class CommonArgs(BaseModel):
    s3_prefix: str = Field(..., description="S3 prefix (e.g. Haryana2)")
    params_path: str = Field("params.yaml", description="Path to params.yaml")
    env_file: Optional[str] = Field(None, description="Env file (defaults inside your code)")
    debug_creds: bool = Field(False, description="Print AWS env presence")
    log_level: str = Field("INFO", description="DEBUG/INFO/WARNING/ERROR")
    upload_log_to_s3: bool = Field(True, description="Upload stage log to S3")
    force: bool = Field(False, description="Re-run even if outputs exist")


class OCRArgs(CommonArgs):
    max_pages: Optional[int] = Field(None, description="Limit pages per PDF")


class FEXTArgs(CommonArgs):
    pass


class CLEANArgs(CommonArgs):
    clean_mapping_path: str = Field(
        "utils/clean/csv/Haryana_Clean_Data_final_with fileNames.xlsx",
        description="Local path to clean mapping file",
    )


class PipelineArgs(CommonArgs):
    max_pages: Optional[int] = Field(None, description="Limit pages per PDF")
    clean_mapping_path: str = Field(
        "utils/clean/csv/Haryana_Clean_Data_final_with fileNames.xlsx",
        description="Local path to clean mapping file",
    )


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/run/ocr")
def run_ocr(req: OCRArgs):
    kwargs = dict(
        s3_prefix=req.s3_prefix,
        params_path=req.params_path,
        env_file=req.env_file,
        max_pages=req.max_pages,
        debug_creds=req.debug_creds,
        log_level=req.log_level,
        upload_log_to_s3=req.upload_log_to_s3,
        force=req.force,
    )
    job_id = _submit_job(run_stage_01_ocr_from_s3, job_name="ocr", kwargs=kwargs)
    return {"job_id": job_id}


@app.post("/run/fext")
def run_fext(req: FEXTArgs):
    kwargs = dict(
        s3_prefix=req.s3_prefix,
        params_path=req.params_path,
        env_file=req.env_file,
        debug_creds=req.debug_creds,
        log_level=req.log_level,
        upload_log_to_s3=req.upload_log_to_s3,
        force=req.force,
    )
    job_id = _submit_job(run_stage_02_fext_from_s3, job_name="fext", kwargs=kwargs)
    return {"job_id": job_id}


@app.post("/run/clean")
def run_clean(req: CLEANArgs):
    kwargs = dict(
        s3_prefix=req.s3_prefix,
        params_path=req.params_path,
        env_file=req.env_file,
        clean_mapping_path=req.clean_mapping_path,
        debug_creds=req.debug_creds,
        log_level=req.log_level,
        upload_log_to_s3=req.upload_log_to_s3,
        force=req.force,
    )
    job_id = _submit_job(run_stage_03_clean_from_s3, job_name="clean", kwargs=kwargs)
    return {"job_id": job_id}


@app.post("/run/pipeline")
def run_pipeline(req: PipelineArgs):
    # One job that runs OCR -> FEXT -> CLEAN sequentially (same as your CLI pipeline flow)
    # See your existing ordering in main.py pipeline branch. 
    # (OCR then FEXT then CLEAN) :contentReference[oaicite:4]{index=4}
    def _pipeline_runner(
        s3_prefix: str,
        params_path: str,
        env_file: Optional[str],
        max_pages: Optional[int],
        clean_mapping_path: str,
        debug_creds: bool,
        log_level: str,
        upload_log_to_s3: bool,
        force: bool,
    ) -> None:
        run_stage_01_ocr_from_s3(
            s3_prefix=s3_prefix,
            params_path=params_path,
            env_file=env_file,
            max_pages=max_pages,
            debug_creds=debug_creds,
            log_level=log_level,
            upload_log_to_s3=upload_log_to_s3,
            force=force,
        )
        run_stage_02_fext_from_s3(
            s3_prefix=s3_prefix,
            params_path=params_path,
            env_file=env_file,
            debug_creds=debug_creds,
            log_level=log_level,
            upload_log_to_s3=upload_log_to_s3,
            force=force,
        )
        run_stage_03_clean_from_s3(
            s3_prefix=s3_prefix,
            params_path=params_path,
            env_file=env_file,
            clean_mapping_path=clean_mapping_path,
            debug_creds=debug_creds,
            log_level=log_level,
            upload_log_to_s3=upload_log_to_s3,
            force=force,
        )

    kwargs = dict(
        s3_prefix=req.s3_prefix,
        params_path=req.params_path,
        env_file=req.env_file,
        max_pages=req.max_pages,
        clean_mapping_path=req.clean_mapping_path,
        debug_creds=req.debug_creds,
        log_level=req.log_level,
        upload_log_to_s3=req.upload_log_to_s3,
        force=req.force,
    )

    job_id = _submit_job(_pipeline_runner, job_name="pipeline", kwargs=kwargs)
    return {"job_id": job_id}
