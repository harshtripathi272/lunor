#!/bin/bash

echo "Setting up NLLB Translation Backend..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run server: python run_server.py"
echo "3. Access API docs at: http://localhost:8000/docs"
