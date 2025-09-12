#!/bin/bash

set -e

echo

echo "📝 Running type checking with mypy ..."
uv run mypy dreadnode
echo "✅ Type checking passed!"
echo

echo "🔎 Running linting with ruff ..."
uv run ruff check dreadnode
echo "✅ Linting passed!"
echo

echo "🎨 Checking formatting with ruff ..."
uv run ruff format --check dreadnode
echo "✅ Code formatting is correct!"
echo

echo "🎉 All checks passed! Code is ready to go."
