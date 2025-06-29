#!/usr/bin/env python3

"""
Setup script for CODY Agent
Helps users install dependencies and configure the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install core dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment configuration."""
    print("\n⚙️ Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example file
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✓ Created .env file from template")
        print("📝 Please edit .env file with your API keys")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("⚠ No .env.example file found")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "cache",
        "tests",
        "examples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/ directory")
    
    return True

def run_tests():
    """Run basic tests to verify installation."""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_cody.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys:")
    print("   - DEEPSEEK_API_KEY (required)")
    print("   - GEMINI_API_KEY (optional)")
    print("   - SERPAPI_KEY (optional)")
    print("\n2. Run CODY:")
    print("   python agent.py")
    print("\n3. Try some commands:")
    print("   /help - Show all commands")
    print("   /add <file> - Add file to context")
    print("   'Create a Python function' - Natural language command")
    print("\n📚 Documentation: README.md")

def main():
    """Main setup function."""
    print("🤖 CODY Agent Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Run tests
    if success and not run_tests():
        print("⚠ Tests failed, but setup may still work")
    
    # Show next steps
    if success:
        show_next_steps()
    else:
        print("\n❌ Setup encountered some issues")
        print("Please check the error messages above and try again")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
