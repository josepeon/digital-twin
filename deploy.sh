#!/usr/bin/env bash
# deploy.sh — Build staging directory and deploy to Railway
#
# Usage:
#   ./deploy.sh           # deploy to Railway
#   ./deploy.sh --dry-run # build staging dir only (no deploy)
#
# This script exists because data/personal/ is gitignored but needed
# at runtime. Railway's GitHub-triggered builds can't access these files,
# so we build a staging directory locally and use `railway up`.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGING_DIR="/tmp/dt-deploy"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "╔══════════════════════════════════════════════╗"
echo "║   Digital Twin — Deploy to Railway           ║"
echo "╚══════════════════════════════════════════════╝"
echo

# ── Clean & create staging dir ─────────────────────────────────────────────────
echo "  [1/4] Building staging directory..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/src"
mkdir -p "$STAGING_DIR/data/personal/rag_index"
mkdir -p "$STAGING_DIR/data/personal/voice"
mkdir -p "$STAGING_DIR/data/personal/identity"

# ── Copy files ─────────────────────────────────────────────────────────────────
echo "  [2/4] Copying files..."

# Application code
cp "$PROJECT_ROOT"/src/*.py "$STAGING_DIR/src/"

# Static files (UI, PWA assets)
if [[ -d "$PROJECT_ROOT/src/static" ]]; then
    cp -r "$PROJECT_ROOT/src/static" "$STAGING_DIR/src/static/"
fi

# Config files
cp "$PROJECT_ROOT/requirements.txt" "$STAGING_DIR/"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGING_DIR/"
cp "$PROJECT_ROOT/Dockerfile" "$STAGING_DIR/"
cp "$PROJECT_ROOT/railway.toml" "$STAGING_DIR/"

# Personal data (gitignored — this is why we need the staging dir)
cp -r "$PROJECT_ROOT/data/personal/rag_index/"* "$STAGING_DIR/data/personal/rag_index/" 2>/dev/null || true
cp "$PROJECT_ROOT/data/personal/voice/voice_id_en.txt" "$STAGING_DIR/data/personal/voice/" 2>/dev/null || true
cp "$PROJECT_ROOT/data/personal/voice/voice_id_es.txt" "$STAGING_DIR/data/personal/voice/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/data/personal/identity/"* "$STAGING_DIR/data/personal/identity/" 2>/dev/null || true

# Minimal .dockerignore
cat > "$STAGING_DIR/.dockerignore" << 'EOF'
__pycache__/
*.py[cod]
.DS_Store
.git/
EOF

# ── Verify ─────────────────────────────────────────────────────────────────────
echo "  [3/4] Verifying staging directory..."
ERRORS=0

for f in src/personality_engine.py src/twin_api.py src/config.py src/voice.py \
         src/memory.py src/static/index.html src/static/manifest.json src/static/sw.js \
         requirements.txt Dockerfile railway.toml pyproject.toml \
         data/personal/identity/jose_facts.json; do
    if [[ ! -f "$STAGING_DIR/$f" ]]; then
        echo "    ❌ Missing: $f"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check RAG index
if [[ ! -d "$STAGING_DIR/data/personal/rag_index" ]] || \
   [[ -z "$(ls -A "$STAGING_DIR/data/personal/rag_index" 2>/dev/null)" ]]; then
    echo "    ❌ RAG index is empty or missing"
    ERRORS=$((ERRORS + 1))
fi

# Check voice IDs
for f in voice_id_en.txt voice_id_es.txt; do
    if [[ ! -f "$STAGING_DIR/data/personal/voice/$f" ]]; then
        echo "    ⚠️  Missing: data/personal/voice/$f (voice will be disabled)"
    fi
done

if [[ $ERRORS -gt 0 ]]; then
    echo "    ❌ $ERRORS critical files missing. Aborting."
    exit 1
fi

SIZE=$(du -sh "$STAGING_DIR" | cut -f1)
echo "    ✅ Staging directory ready ($SIZE)"

# ── Deploy ─────────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "  [4/4] Dry run — skipping deploy."
    echo "    Staging dir: $STAGING_DIR"
    exit 0
fi

echo "  [4/4] Deploying to Railway..."
cd "$STAGING_DIR"
railway up --detach

echo
echo "  ✅ Deploy initiated! Check status with:"
echo "     railway deployment list"
echo "     railway logs"
