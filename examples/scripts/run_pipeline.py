#!/usr/bin/env python3
"""
Example script for running an ML pipeline.

This script demonstrates how to use the Autonomous ML Agent
to create and execute a machine learning pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.agent import AutonomousMLAgent
from config.settings import get_settings

async def main():
    """Main function to run the example pipeline."""
    print("🚀 Starting Autonomous ML Agent Example")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Initialize agent
        agent = AutonomousMLAgent(settings)
        await agent.initialize()
        
        print("✅ Agent initialized successfully")
        
        # Example dataset path (you would need to provide a real dataset)
        dataset_path = "examples/sample_datasets/iris.csv"
        
        if not Path(dataset_path).exists():
            print(f"⚠️  Dataset not found at {dataset_path}")
            print("Please provide a valid dataset path")
            return
        
        # Run pipeline
        print("🚀 Creating ML pipeline...")
        pipeline_id = await agent.run_pipeline(
            dataset_path=dataset_path,
            target_column="target",
            task_type="classification",
            optimization_metric="accuracy"
        )
        
        print(f"✅ Pipeline created with ID: {pipeline_id}")
        
        # Monitor pipeline
        print("📊 Monitoring pipeline progress...")
        while True:
            status = await agent.get_pipeline_status(pipeline_id)
            print(f"Status: {status.get('status', 'Unknown')}")
            
            if status.get('status') in ['completed', 'failed']:
                break
            
            await asyncio.sleep(5)
        
        if status.get('status') == 'completed':
            print("🎉 Pipeline completed successfully!")
            
            # Get results
            results = await agent.get_pipeline_results(pipeline_id)
            print(f"Best model accuracy: {results.get('best_model', {}).get('accuracy', 'N/A')}")
        else:
            print("❌ Pipeline failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        if 'agent' in locals():
            await agent.cleanup()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
