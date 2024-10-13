#!/bin/bash

cd crowndata-evaluation
pip install poetry
poetry install
poetry run uvicorn crowndata_evaluation.main:app --host 0.0.0.0 --port 8000