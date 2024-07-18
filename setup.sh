#!/bin/bash

# Update package lists
sudo apt-get update

# Install build-essential and other required packages
sudo apt-get install -y build-essential

# Install Python packages
pip install -r requirements.txt
