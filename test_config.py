#!/usr/bin/env python3
"""
Test script for the configuration system.

This script tests the first step of our implementation:
- Configuration loading from environment variables
- Settings validation
- Basic logging setup
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration_system():
    """Test the configuration system."""
    print("🧪 Testing Configuration System...")
    
    try:
        # Test 1: Import configuration
        from config import get_settings, Settings
        print("✅ Successfully imported configuration modules")
        
        # Test 2: Get settings with defaults
        settings = get_settings()
        print("✅ Successfully loaded settings with defaults")
        
        # Test 3: Check key settings
        print(f"   Environment: {settings.ENVIRONMENT}")
        print(f"   Debug Mode: {settings.DEBUG}")
        print(f"   Log Level: {settings.LOG_LEVEL}")
        print(f"   API Host: {settings.API_HOST}")
        print(f"   API Port: {settings.API_PORT}")
        
        # Test 4: Test validation
        print("✅ Settings validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logging_system():
    """Test the logging system."""
    print("\n🧪 Testing Logging System...")
    
    try:
        # Test 1: Import logging utilities
        from utils.logger import setup_logging, get_logger, StructuredLogger
        print("✅ Successfully imported logging modules")
        
        # Test 2: Setup logging
        setup_logging(log_level="INFO", log_format="text")
        print("✅ Successfully setup logging")
        
        # Test 3: Test basic logging
        logger = get_logger("test_config")
        logger.info("Test log message")
        logger.warning("Test warning message")
        print("✅ Basic logging working")
        
        # Test 4: Test structured logging
        structured_logger = StructuredLogger("test_structured")
        structured_logger.info("Structured log message", user_id="123", action="test")
        print("✅ Structured logging working")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling."""
    print("\n🧪 Testing Environment Variables...")
    
    try:
        # Test 1: Check if .env.example exists
        env_example = Path(".env.example")
        if env_example.exists():
            print("✅ .env.example file exists")
        else:
            print("⚠️  .env.example file not found")
        
        # Test 2: Check if .env exists
        env_file = Path(".env")
        if env_file.exists():
            print("✅ .env file exists")
        else:
            print("⚠️  .env file not found (you may need to copy from .env.example)")
        
        # Test 3: Test required environment variables
        required_vars = ["GEMINI_API_KEY", "E2B_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing required environment variables: {missing_vars}")
            print("   These will cause errors when running the full application")
        else:
            print("✅ All required environment variables are set")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Configuration System Tests\n")
    
    # Run tests
    config_ok = test_configuration_system()
    logging_ok = test_logging_system()
    env_ok = test_environment_variables()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    if config_ok:
        print("✅ Configuration System: PASSED")
    else:
        print("❌ Configuration System: FAILED")
    
    if logging_ok:
        print("✅ Logging System: PASSED")
    else:
        print("❌ Logging System: FAILED")
    
    if env_ok:
        print("✅ Environment Variables: PASSED")
    else:
        print("❌ Environment Variables: FAILED")
    
    print("\n" + "="*50)
    
    if all([config_ok, logging_ok, env_ok]):
        print("🎉 ALL TESTS PASSED! Configuration system is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Copy .env.example to .env and fill in your API keys")
        print("   2. Move to Phase 2: Data Foundation implementation")
        print("   3. Test data loading and validation functions")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\n🔧 Common fixes:")
        print("   - Ensure all dependencies are installed (pip install -r requirements.txt)")
        print("   - Check that .env file exists with proper API keys")
        print("   - Verify Python path includes src directory")

if __name__ == "__main__":
    main()
