#!/bin/bash
# Audio Ops Framework - Complete Setup Script
# Handles: directories, configuration, and Python environment (Poetry)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Audio Ops Framework - Complete Setup                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 1: Prerequisites Check
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 1: Checking prerequisites${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC} Docker not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker installed"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗${NC} Docker Compose not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker Compose installed"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗${NC} Python 3 not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 3 installed"

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 2: Create Directory Structure
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 2: Creating directory structure${NC}"

DIRS=(
    "config"
    "dagster_home"
    "mlflow"
    "mlflow/artifacts"
    "label_studio_data"
    "minio_storage"
    "influxdb_data"
    "grafana_data"
    "demo_audio"
    "logs"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓${NC} $dir"
    fi
done

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 3: Setup Configuration Files
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 3: Setting up configuration files${NC}"

# .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created .env"
    else
        echo -e "${RED}✗${NC} .env.example not found"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} .env already exists (skipping)"
fi

# .secrets.env
if [ ! -f ".secrets.env" ]; then
    if [ -f ".secrets.env.example" ]; then
        cp .secrets.env.example .secrets.env
        chmod 600 .secrets.env
        echo -e "${GREEN}✓${NC} Created .secrets.env (protected)"
    else
        echo -e "${RED}✗${NC} .secrets.env.example not found"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} .secrets.env already exists (skipping)"
fi

# config files
if [ ! -f "config/config.yaml" ] && [ -f "config.yaml" ]; then
    cp config.yaml config/config.yaml
    echo -e "${GREEN}✓${NC} Copied config.yaml to config/"
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 4: Setup .gitignore
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 4: Configuring .gitignore${NC}"

GITIGNORE_ENTRIES=(
    ".env"
    ".secrets.env"
    "secrets/"
    ".venv/"
    ".pytest_cache/"
    "__pycache__/"
    "*.pyc"
    "dist/"
    "build/"
    "*.egg-info/"
    ".DS_Store"
    "dagster_home/"
    "mlflow/"
    "minio_storage/"
    "influxdb_data/"
    "grafana_data/"
    "label_studio_data/"
)

if [ ! -f ".gitignore" ]; then
    touch .gitignore
fi

for entry in "${GITIGNORE_ENTRIES[@]}"; do
    if ! grep -q "^${entry}$" .gitignore; then
        echo "$entry" >> .gitignore
    fi
done
echo -e "${GREEN}✓${NC} .gitignore configured"

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 5: Install Poetry
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 5: Setting up Poetry environment${NC}"

POETRY_PATH="$HOME/.local/bin"

if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add to PATH for current session
    export PATH="$POETRY_PATH:$PATH"
    
    # Add to shell config for future sessions
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        fi
    fi
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.zshrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
        fi
    fi
    
    if command -v poetry &> /dev/null; then
        echo -e "${GREEN}✓${NC} Poetry installed"
    else
        echo -e "${RED}✗${NC} Poetry installation failed"
        exit 1
    fi
else
    POETRY_VERSION=$(poetry --version)
    echo -e "${GREEN}✓${NC} Poetry already installed ($POETRY_VERSION)"
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Step 6: Install Python Dependencies
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}Step 6: Installing Python dependencies${NC}"

if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}✗${NC} pyproject.toml not found"
    exit 1
fi

poetry install
echo -e "${GREEN}✓${NC} Dependencies installed"

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Setup Complete!                                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Restart your shell (or source config):"
echo "   bash          # New bash session"
echo "   # OR manually:"
echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "2. Activate Python environment:"
echo "   eval \"\$(poetry env activate)\""
echo ""
echo "   Or manually:"
echo "   source \$(poetry env info --path)/bin/activate"
echo ""
echo "3. Review and update configuration:"
echo "   nano .env              # Review environment settings"
echo "   nano .secrets.env      # Change passwords (REQUIRED)"
echo ""
echo "4. Start Docker services:"
echo "   docker-compose --env-file .env up -d"
echo ""
echo "5. Verify setup:"
echo "   make health"
echo ""
echo "6. Run tests (optional):"
echo "   make test"
echo ""

echo -e "${YELLOW}Important:${NC}"
echo "  • Edit .secrets.env and change all default passwords"
echo "  • Never commit .env or .secrets.env to git"
echo "  • Poetry was installed to: $POETRY_PATH"
echo ""

echo -e "${YELLOW}Documentation:${NC}"
echo "  • QUICK_START.md - Quick reference"
echo "  • CONFIG_README.md - Configuration details"
echo "  • Makefile - Available commands (run: make help)"
echo ""