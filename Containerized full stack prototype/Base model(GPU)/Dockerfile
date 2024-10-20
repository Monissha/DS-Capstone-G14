# Use NVIDIA CUDA image with full runtime support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies including Python, pip, build tools, CUDA, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8 \
    python3 python3-pip build-essential wget git && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade SQLite to version 3.42.0 or later
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3410000.tar.gz && \
    tar xvfz sqlite-autoconf-3410000.tar.gz && \
    cd sqlite-autoconf-3410000 && \
    ./configure && make && make install && \
    rm -rf /sqlite-autoconf-3410000*

# Set environment variables for the updated SQLite
ENV PATH="/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install the packaging module
RUN pip install packaging

# Install Flash Attention
RUN pip install flash-attn==2.6.3

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
