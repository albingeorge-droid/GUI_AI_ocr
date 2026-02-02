import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "pgchs-ocr"

list_of_files = [
    "artifacts/pdfs/.gitkeep",
    "artifacts/chroma/.gitkeep",    
    "artifacts/results/.gitkeep",
    
    "notebooks/.gitkeep",

    # pipeline
    "ocr_pipeline/__init__.py",
    "ocr_pipeline/stage_01_data_ingestion.py",
    "ocr_pipeline/stage_02_data_processing.py",
    "ocr_pipeline/stage_03_ocr.py",
    "ocr_pipeline/stage_04_pattern_matching.py",
    "ocr_pipeline/stage_05_evaluation.py",

    # utils
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    "utils/server.py",
    "utils/common.py",
    
    "utils/llm/__init__.py",
    "utils/llm/models.py",
    "utils/llm/prompt_temps.py",
    
    "utils/ocr/__init__.py",

    # main
    # "main.py",
    "app.py",

    # tests
    "tests/__init__.py",

    "params.yaml",
    "dvc.yaml",
    ".env.local",
    ".env.example",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")