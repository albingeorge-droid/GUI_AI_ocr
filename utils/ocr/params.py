from dataclasses import dataclass
from typing import Any
import yaml

@dataclass(frozen=True)
class OCRConfig:
    bucket: str
    region: str
    model_id: str
    dpi: int
    temperature: float
    max_tokens: int
    retries: int

@dataclass(frozen=True)
class PhoenixConfig:
    otlp_endpoint: str | None
    service_name: str

def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_ocr_config(params: dict[str, Any]) -> OCRConfig:
    o = (params.get("ocr") or {})
    return OCRConfig(
        bucket=o["bucket"],
        region=o.get("region", "us-east-1"),
        model_id=o.get("model_id", "us.meta.llama4-scout-17b-instruct-v1:0"),
        dpi=int(o.get("dpi", 250)),
        temperature=float(o.get("temperature", 0.0)),
        max_tokens=int(o.get("max_tokens", 2500)),
        retries=int(o.get("retries", 6)),
    )

def get_phoenix_config(params: dict[str, Any]) -> PhoenixConfig:
    p = (params.get("phoenix") or {})
    endpoint = p.get("otlp_endpoint", "http://localhost:4317")
    if endpoint is not None:
        endpoint = str(endpoint).strip()
        if endpoint == "":
            endpoint = None
    return PhoenixConfig(
        otlp_endpoint=endpoint,
        service_name=str(p.get("service_name", "bedrock-pdf-ocr")),
    )
