# Use a base image with Python and no GPU dependencies
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies including Python, pip, build tools, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget git && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install necessary packages including PyTorch for CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the packaging module
RUN pip install packaging

# Install Flash Attention only if you decide to use it later for CPU
# Remove this line if you do not need it
# RUN pip install flash-attn==2.6.3

# Copy the local files into the container's working directory
COPY . /app

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# Install necessary packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your pre-trained vector embeddings
COPY ./vector1 /app/vector1

# Create a symbolic link for Python to make it accessible as 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Expose the port Gradio/Flask will run on
EXPOSE 7860

# Command to run the Flask application along with the Gradio interface
CMD ["python", "app.py"]
