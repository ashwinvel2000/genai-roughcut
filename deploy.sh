#!/bin/bash

# Azure Container Apps Deployment Script
# Usage: ./deploy.sh

set -e

# Configuration
RESOURCE_GROUP="ai-video-generator-rg"
LOCATION="eastus"
ACR_NAME="aivideogenacr"  # Change this to something unique
CONTAINER_APP_ENV="ai-video-env"
CONTAINER_APP_NAME="ai-video-generator"
IMAGE_NAME="ai-video-generator"

echo "=== Azure Container Apps Deployment ==="
echo ""
echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  ACR Name: $ACR_NAME"
echo "  App Name: $CONTAINER_APP_NAME"
echo ""

# Check if logged in
echo "Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure first:"
    az login
fi

# Create resource group
echo ""
echo "Creating resource group..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --output table

# Create Azure Container Registry
echo ""
echo "Creating Azure Container Registry..."
if az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "ACR already exists, skipping..."
else
    az acr create \
      --resource-group $RESOURCE_GROUP \
      --name $ACR_NAME \
      --sku Basic \
      --admin-enabled true \
      --output table
fi

# Get ACR credentials
echo ""
echo "Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

echo "ACR Login Server: $ACR_LOGIN_SERVER"

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t $IMAGE_NAME:latest .

# Login to ACR
echo ""
echo "Logging into ACR..."
az acr login --name $ACR_NAME

# Tag and push image
echo ""
echo "Tagging and pushing image to ACR..."
docker tag $IMAGE_NAME:latest $ACR_LOGIN_SERVER/$IMAGE_NAME:latest
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:latest

# Create Container Apps environment
echo ""
echo "Creating Container Apps environment..."
if az containerapp env show --name $CONTAINER_APP_ENV --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Environment already exists, skipping..."
else
    az containerapp env create \
      --name $CONTAINER_APP_ENV \
      --resource-group $RESOURCE_GROUP \
      --location $LOCATION \
      --output table
fi

# Deploy or update container app
echo ""
if az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Updating existing container app..."
    az containerapp update \
      --name $CONTAINER_APP_NAME \
      --resource-group $RESOURCE_GROUP \
      --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
      --output table
else
    echo "Creating new container app..."
    az containerapp create \
      --name $CONTAINER_APP_NAME \
      --resource-group $RESOURCE_GROUP \
      --environment $CONTAINER_APP_ENV \
      --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
      --registry-server $ACR_LOGIN_SERVER \
      --registry-username $ACR_USERNAME \
      --registry-password $ACR_PASSWORD \
      --target-port 8080 \
      --ingress external \
      --min-replicas 1 \
      --max-replicas 3 \
      --cpu 1.0 \
      --memory 2.0Gi \
      --env-vars LOG_LEVEL=INFO \
      --output table
fi

# Get application URL
echo ""
echo "=== Deployment Complete! ==="
echo ""
APP_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  -o tsv)

echo "Your application is available at:"
echo "  https://$APP_URL"
echo ""
echo "Health check:"
echo "  curl https://$APP_URL/healthz"
echo ""
echo "To view logs:"
echo "  az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --tail 100 --follow"
echo ""
