#!/bin/bash

# Install system dependencies for Tesseract OCR
apt-get update && apt-get install -y tesseract-ocr

# Start the Flask application
python app.py
