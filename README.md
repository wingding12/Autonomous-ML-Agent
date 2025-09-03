# Autonomous ML Agent

An intelligent, end-to-end machine learning pipeline orchestrated by Large Language Models (LLMs) that automatically handles data ingestion, preprocessing, model training, optimization, and deployment.

## 🚀 Features

- **Autonomous Pipeline Orchestration**: LLM-driven decision making for ML workflows
- **Intelligent Data Preprocessing**: Automatic handling of missing values, categorical encoding, and feature engineering
- **Meta-Learning Optimization**: Uses prior experiment metadata to warm-start hyperparameter search
- **Multi-Model Support**: Logistic/Linear Regression, Random Forest, Gradient Boosting, kNN, MLP
- **Advanced Optimization**: Bayesian and random search with intelligent initialization
- **Interactive Leaderboard**: Streamlit-based UI showing model performance and comparisons
- **Production Ready**: Export pipelines as FastAPI services with auto-generated deployment scripts
- **Ensemble Methods**: LLM-proposed ensemble strategies for optimal performance

## 🏗️ Architecture

The system consists of several key components:

- **Agent Core**: Orchestrates LLM decisions and manages pipeline lifecycle
- **LLM Integration**: Google Gemini API with code execution capabilities
- **Sandbox Execution**: E2B sandbox for isolated, secure code execution
- **ML Pipeline**: Comprehensive ML workflow from preprocessing to deployment
- **Meta-Learning**: Intelligent hyperparameter optimization using historical data
- **API Service**: FastAPI backend for pipeline management and inference

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI
- **LLM**: Google Gemini API
- **Code Execution**: E2B Sandbox
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Optimization**: Optuna
- **UI**: Streamlit
- **Deployment**: Docker, FastAPI

## 📦 Installation

### Prerequisites

- Python 3.9+
- Docker (for deployment)
- Google Gemini API key
- E2B API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run the application:

```bash
# Start FastAPI backend
uvicorn src.main:app --reload

# Start Streamlit UI (in another terminal)
streamlit run src/ui/streamlit_app.py
```

## 🚀 Quick Start

### Basic Usage

```python
from src.core.agent import AutonomousMLAgent

# Initialize the agent
agent = AutonomousMLAgent()

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

### API Usage

```bash
# Create a new pipeline
curl -X POST "http://localhost:8000/api/v1/pipeline/create" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_path": "data/iris.csv",
       "target_column": "target",
       "task_type": "classification"
     }'

# Get pipeline status
curl "http://localhost:8000/api/v1/pipeline/{pipeline_id}/status"
```

## 📊 Example Workflows

### Classification Task

1. Upload dataset (e.g., Iris, Titanic)
2. Agent automatically detects data types and issues
3. LLM generates preprocessing pipeline
4. Multiple models trained with hyperparameter optimization
5. Best model selected and ensemble strategies evaluated
6. Pipeline exported as deployable service

### Regression Task

1. Dataset analysis for numerical targets
2. Feature engineering and scaling
3. Model training with cross-validation
4. Performance comparison and selection
5. Model interpretation and feature importance

## 🔧 Configuration

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
MODEL_CACHE_DIR=./cache/models
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

## 📈 Performance

- **Pipeline Creation**: 2-5 minutes for typical datasets
- **Model Training**: 1-10 minutes per model (depending on dataset size)
- **Hyperparameter Optimization**: 5-30 minutes with meta-learning warm-starts
- **Total Pipeline Time**: 15-60 minutes for complete workflows

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
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

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

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
