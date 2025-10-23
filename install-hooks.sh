#!/bin/sh
# Install git hooks from the hooks/ directory

HOOKS_DIR="$(cd "$(dirname "$0")/hooks" && pwd)"
GIT_HOOKS_DIR="$(cd "$(dirname "$0")/.git/hooks" && pwd)"

echo "Installing git hooks..."

for hook in "$HOOKS_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        ln -sf "$HOOKS_DIR/$hook_name" "$GIT_HOOKS_DIR/$hook_name"
        echo "✅ Installed $hook_name"
    fi
done

echo "Git hooks installation complete!"
