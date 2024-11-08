# Use an official Python 3.10 slim-based image
FROM python:3.10-slim-buster

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN pip install --upgrade pip

# Install Concrete-ML and dependencies
RUN pip install concrete-ml pandas scikit-learn jupyter

# Create a working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port for Jupyter notebook
EXPOSE 8888

# Set the command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
