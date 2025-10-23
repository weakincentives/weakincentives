#!/usr/bin/env bash
set -euo pipefail

# Script to bump the version number in pyproject.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYPROJECT="$SCRIPT_DIR/pyproject.toml"

usage() {
    echo "Usage: $0 [major|minor|patch|VERSION]"
    echo ""
    echo "Examples:"
    echo "  $0 patch      # 0.1.0 -> 0.1.1"
    echo "  $0 minor      # 0.1.0 -> 0.2.0"
    echo "  $0 major      # 0.1.0 -> 1.0.0"
    echo "  $0 1.2.3      # Set version to 1.2.3"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

# Get current version
CURRENT_VERSION=$(grep '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: Could not find version in $PYPROJECT"
    exit 1
fi

echo "Current version: $CURRENT_VERSION"

# Parse version
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Calculate new version
case "$1" in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    *)
        # Assume it's a specific version number
        if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Error: Invalid version format. Use major.minor.patch (e.g., 1.2.3)"
            exit 1
        fi
        NEW_VERSION="$1"
        ;;
esac

echo "New version: $NEW_VERSION"

# Update pyproject.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"
else
    # Linux
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"
fi

echo "✅ Updated $PYPROJECT to version $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. Review the change: git diff pyproject.toml"
echo "  2. Commit: git commit -am 'Bump version to $NEW_VERSION'"
echo "  3. Tag: git tag v$NEW_VERSION"
echo "  4. Push: git push && git push --tags"
