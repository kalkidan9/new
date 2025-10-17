#!/bin/bash

# Install system dependencies for Tesseract OCR
apt-get update && apt-get install -y tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask application
python3 app.py
