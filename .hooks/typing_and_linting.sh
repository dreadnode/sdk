#!/bin/bash

set -e

echo

echo "📝 Running type checking with mypy ..."
uv run mypy dreadnode
echo "✅ Type checking passed!"
echo

echo "🔎 Running linting with ruff ..."
uv run ruff check --output-format=github --fix .
echo "✅ Linting passed!"
echo

echo "🎨 Formatting code with ruff ..."
uv run ruff format .
echo "✅ Code formatted!"
echo

echo "🎉 All checks passed! Code is ready to go."
