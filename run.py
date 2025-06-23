import os
import subprocess
import sys
import webbrowser
from time import sleep

def check_env_file():
    """Check if .env file exists and create from template if not."""
    if not os.path.exists("backend/.env"):
        if os.path.exists("backend/.env.template"):
            print("⚠️ No .env file found. Creating from template...")
            with open("backend/.env.template", "r") as template:
                with open("backend/.env", "w") as env_file:
                    env_file.write(template.read())
            print("✅ Created .env file from template. Please edit backend/.env to add your API key.")
            print("   Get your API key from: https://makersuite.google.com/")
            return False
        else:
            print("❌ Error: Could not find .env.template file.")
            return False
    return True

def run_app():
    """Run the Streamlit app."""
    try:
        # Check if Streamlit is installed
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "streamlit"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        print("🚀 Starting DocuMind application...")
        print("📊 Streamlit server starting...")
        
        # Open browser after a short delay
        def open_browser():
            sleep(2)  # Wait for Streamlit to start
            webbrowser.open("http://localhost:8501")
            print("🌐 Opening browser at http://localhost:8501")
        
        # Start browser in a separate thread
        import threading
        threading.Thread(target=open_browser).start()
        
        # Run Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("❌ Error: Streamlit not found. Please install requirements first:")
        print("   pip install -r backend/requirements.txt")
        return
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")

if __name__ == "__main__":
    # Check if requirements are installed
    if not os.path.exists("backend/__pycache__"):
        print("⚠️ First run detected. Checking requirements...")
        try:
            print("📦 Installing requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"],
                check=True,
            )
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error installing requirements.")
            sys.exit(1)
    
    # Check environment file
    env_ready = check_env_file()
    
    if env_ready:
        # Run the application
        run_app()
    else:
        print("\n🔑 Please edit the backend/.env file to add your API key before running again.")
        print("   Then run this script again to start the application.")
