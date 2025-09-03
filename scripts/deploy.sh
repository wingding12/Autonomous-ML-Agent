#!/bin/bash

# Autonomous ML Agent Deployment Script
# This script deploys the application to different environments

set -e

# Default values
ENVIRONMENT=${1:-development}
DOCKER_COMPOSE_FILE="docker-compose.yml"

echo "🚀 Deploying Autonomous ML Agent to $ENVIRONMENT environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

case $ENVIRONMENT in
    "development")
        echo "🔧 Deploying to development environment..."
        
        # Build and start services
        docker-compose -f $DOCKER_COMPOSE_FILE up --build -d
        
        echo "✅ Development deployment completed!"
        echo ""
        echo "Services available at:"
        echo "- API: http://localhost:8000"
        echo "- UI: http://localhost:8501"
        echo "- MLflow: http://localhost:5000"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
    
    "production")
        echo "🚀 Deploying to production environment..."
        
        # Check if .env.production exists
        if [ ! -f ".env.production" ]; then
            echo "❌ Production environment file (.env.production) not found."
            echo "Please create it with production configuration."
            exit 1
        fi
        
        # Use production environment file
        export ENV_FILE=.env.production
        
        # Build and start production services
        docker-compose -f $DOCKER_COMPOSE_FILE --env-file .env.production up --build -d
        
        echo "✅ Production deployment completed!"
        echo ""
        echo "Services deployed with production configuration"
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
    
    "staging")
        echo "🔍 Deploying to staging environment..."
        
        # Check if .env.staging exists
        if [ ! -f ".env.staging" ]; then
            echo "❌ Staging environment file (.env.staging) not found."
            echo "Please create it with staging configuration."
            exit 1
        fi
        
        # Use staging environment file
        export ENV_FILE=.env.staging
        
        # Build and start staging services
        docker-compose -f $DOCKER_COMPOSE_FILE --env-file .env.staging up --build -d
        
        echo "✅ Staging deployment completed!"
        echo ""
        echo "Services deployed with staging configuration"
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
    
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "Supported environments: development, staging, production"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment completed successfully!"
echo "Environment: $ENVIRONMENT"
echo "Timestamp: $(date)"
