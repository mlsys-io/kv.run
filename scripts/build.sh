#!/bin/bash

# Build script for MLOC Docker images

set -e

echo "🏗️  Building MLOC Docker images..."

# Build base image
echo "📦 Building base image..."
docker build -t mloc:base -f docker/base/Dockerfile .

# Build application image
echo "📦 Building application image..."
docker build -t mloc:app -f docker/app/Dockerfile .

echo "✅ Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep mloc