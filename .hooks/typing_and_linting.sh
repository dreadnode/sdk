#!/bin/bash

set -e

poetry run mypy .
poetry run ruff check .
poetry run ruff format --check .
