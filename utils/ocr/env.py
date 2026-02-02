from pathlib import Path
from dotenv import load_dotenv

def load_env(env_file: str | None) -> None:
    """
    Load environment variables from a .env file.
    Defaults to .env.local located next to repo root (or current working dir).
    """
    if env_file:
        load_dotenv(env_file, override=False)
        return

    # Prefer .env.local in repo root (two levels up from utils/ocr)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".env.local"
    if candidate.exists():
        load_dotenv(candidate.as_posix(), override=False)
        return

    # Fallback: .env.local in current working directory
    load_dotenv(".env.local", override=False)
