# Autonomous ML Agent Documentation

Welcome to the documentation for the Autonomous ML Agent project. This comprehensive guide will help you understand, install, and use the system.

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

The Autonomous ML Agent is an intelligent, end-to-end machine learning pipeline orchestrated by Large Language Models (LLMs). It automatically handles data ingestion, preprocessing, model training, optimization, and deployment.

### Key Features

- **Autonomous Pipeline Orchestration**: LLM-driven decision making for ML workflows
- **Intelligent Data Preprocessing**: Automatic handling of missing values, categorical encoding, and feature engineering
- **Meta-Learning Optimization**: Uses prior experiment metadata to warm-start hyperparameter search
- **Multi-Model Support**: Comprehensive ML algorithm support with ensemble methods
- **Interactive UI**: Streamlit-based interface for pipeline management and monitoring
- **Production Ready**: Export pipelines as deployable services

## 🏗️ Architecture

The system consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  Autonomous     │
│                 │◄──►│                 │◄──►│  ML Agent      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   E2B Sandbox   │    │   Google Gemini │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
```

### Component Details

- **Agent Core**: Orchestrates LLM decisions and manages pipeline lifecycle
- **LLM Integration**: Google Gemini API with code execution capabilities
- **Sandbox Execution**: E2B sandbox for isolated, secure code execution
- **ML Pipeline**: Comprehensive ML workflow from preprocessing to deployment
- **Meta-Learning**: Intelligent hyperparameter optimization using historical data

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Docker (for deployment)
- Google Gemini API key
- E2B API key

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Autonomous-ML-Agent.git
   cd Autonomous-ML-Agent
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

## 🎯 Quick Start

### Basic Usage

```python
from src.core.agent import AutonomousMLAgent

# Initialize the agent
agent = AutonomousMLAgent(settings)
await agent.initialize()

# Run a complete ML pipeline
pipeline_id = await agent.run_pipeline(
    dataset_path="data/iris.csv",
    target_column="target",
    task_type="classification",
    optimization_metric="accuracy"
)

# Get results
results = await agent.get_pipeline_results(pipeline_id)
print(f"Best model accuracy: {results.best_model.accuracy}")
```

### Web Interface

1. **Start the API backend**

   ```bash
   uvicorn src.main:app --reload
   ```

2. **Start the Streamlit UI**

   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 📖 API Reference

### Pipeline Management

- `POST /api/v1/pipeline/create` - Create new ML pipeline
- `GET /api/v1/pipeline/{pipeline_id}/status` - Get pipeline status
- `GET /api/v1/pipeline/{pipeline_id}/results` - Get pipeline results
- `DELETE /api/v1/pipeline/{pipeline_id}` - Stop pipeline

### Model Management

- `GET /api/v1/models/leaderboard` - Get model leaderboard
- `GET /api/v1/models/{model_id}/details` - Get model details

### Inference

- `POST /api/v1/inference/single` - Single prediction
- `POST /api/v1/inference/batch` - Batch prediction

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash-exp

# E2B Configuration
E2B_API_KEY=your_e2b_api_key

# Application Configuration
LOG_LEVEL=INFO
MAX_SANDBOX_RUNTIME=3600
```

### Pipeline Configuration

```python
pipeline_config = {
    "max_runtime": 3600,  # seconds
    "max_models": 10,
    "optimization_metric": "accuracy",
    "cross_validation_folds": 5,
    "enable_meta_learning": True,
    "ensemble_strategies": ["stacking", "voting"]
}
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t autonomous-ml-agent .
docker run -p 8000:8000 autonomous-ml-agent
```

### Production Deployment

```bash
# Deploy to production
./scripts/deploy.sh production

# Scale horizontally
docker-compose up --scale api=3
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Install development dependencies**

   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**

   ```bash
   pytest tests/
   ```

3. **Code formatting**
   ```bash
   black src/
   isort src/
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [E2B](https://e2b.dev/) for secure code execution sandboxes
- [Google Gemini](https://ai.google.dev/gemini-api/docs/code-execution) for LLM capabilities
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Optuna](https://optuna.org/) for hyperparameter optimization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Autonomous-ML-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Autonomous-ML-Agent/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/Autonomous-ML-Agent/wiki)

---

**Built with ❤️ by the Autonomous ML Team**
