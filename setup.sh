#!/bin/bash
set -e

echo "🎙️ YouTube Podcast Processor — Setup"
echo "======================================"

# Check Python — skip Windows Store shims by actually running python
if python3 --version &>/dev/null 2>&1; then
    PYTHON=python3
elif python --version &>/dev/null 2>&1; then
    PYTHON=python
else
    echo "❌ Python not found. Install it from https://python.org"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "❌ ffmpeg not found."
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS:         brew install ffmpeg"
    echo "   Windows:       https://ffmpeg.org/download.html"
    exit 1
fi

echo "✅ Python and ffmpeg found"

# Create venv
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate — Windows uses Scripts/, Linux/macOS uses bin/
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install deps
echo "→ Installing dependencies..."
$PYTHON -m pip install -r requirements.txt -q

# Install correct torch build based on CUDA version
echo "→ Checking for NVIDIA GPU..."
if command -v nvidia-smi &>/dev/null; then
    # Extract CUDA version from nvidia-smi (e.g. "12.8" or "13.2")
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "✅ NVIDIA GPU detected — CUDA $CUDA_VERSION"

    # Pick torch index based on CUDA version
    if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
        TORCH_INDEX="cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="cu124"
    else
        TORCH_INDEX="cu121"
    fi

    echo "→ Installing torch for $TORCH_INDEX..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/$TORCH_INDEX --force-reinstall -q
    echo "✅ CUDA torch ($TORCH_INDEX) installed"
else
    echo "⚠️  No NVIDIA GPU detected — using CPU torch (transcription will be slower)"
fi

echo ""

# .env setup
if [ ! -f ".env" ]; then
    echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
    echo "⚠️  Created .env — add your Anthropic API key to it:"
    echo "    https://console.anthropic.com/keys"
else
    echo "✅ .env already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run:"
echo "  Windows (Git Bash): source .venv/Scripts/activate"
echo "  Linux/macOS:        source .venv/bin/activate"
echo "  Then: streamlit run podcast_processor.py"
