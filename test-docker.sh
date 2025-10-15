#!/bin/bash

# Local Docker Testing Script
# Usage: ./test-docker.sh

set -e

IMAGE_NAME="ai-video-generator"

echo "=== Building Docker Image ==="
docker build -t $IMAGE_NAME:latest .

echo ""
echo "=== Starting Container ==="
docker run -d \
  --name ai-video-test \
  -p 8080:8080 \
  $IMAGE_NAME:latest

echo ""
echo "Container started. Waiting for application to be ready..."
sleep 5

echo ""
echo "=== Testing Health Endpoint ==="
if curl -f http://localhost:8080/healthz; then
    echo ""
    echo "✅ Health check passed!"
else
    echo ""
    echo "❌ Health check failed"
    exit 1
fi

echo ""
echo "=== Application is running! ==="
echo "Open your browser to: http://localhost:8080"
echo ""
echo "To view logs:"
echo "  docker logs -f ai-video-test"
echo ""
echo "To stop and remove container:"
echo "  docker stop ai-video-test && docker rm ai-video-test"
echo ""
