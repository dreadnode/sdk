#!/bin/bash

# Get pre-merge hash from the target branch
old_hash=$(git show ORIG_HEAD:uv.lock | md5sum 2> /dev/null || echo "")

# Get current hash
new_hash=$(md5sum uv.lock 2> /dev/null || echo "")

# Compare and run uv sync if changed
if [ "$old_hash" != "$new_hash" ]; then
    echo "📦 Dependencies changed. Running uv sync..."
    uv sync --all-extras || {
        echo "❌ Failed to update dependencies"
        exit 1
    }
    echo "✅ Dependencies updated!"
else
    echo "📦 No dependency changes"
fi
