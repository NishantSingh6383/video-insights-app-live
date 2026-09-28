import subprocess
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, 'backend')

print("=" * 50)
print("VIDEO INSIGHTS APP SETUP")
print("=" * 50)

# Backend setup (frontend is served by Django - no npm step needed)
print("\n[1/2] Setting up backend virtual environment...")
venv_path = os.path.join(backend_dir, 'venv')
if not os.path.exists(venv_path):
    subprocess.run([sys.executable, '-m', 'venv', venv_path], cwd=backend_dir)
    print("  Created venv")
else:
    print("  venv already exists")

print("\n[2/2] Installing backend dependencies...")
if os.name == 'nt':
    pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
else:
    pip_path = os.path.join(venv_path, 'bin', 'pip')
result = subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], cwd=backend_dir, capture_output=True, text=True)
if result.returncode == 0:
    print("  Backend dependencies installed successfully")
else:
    print(f"  Error: {result.stderr}")

print("\n" + "=" * 50)
print("SETUP COMPLETE!")
print("=" * 50)
print("\nTo run the app:")
print("  1. Run start-backend.bat (or: cd backend && python manage.py runserver)")
print("  2. Open http://127.0.0.1:8000")
print("\nOptional: set ANTHROPIC_API_KEY to enable AI insights.")
