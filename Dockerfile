# Use an official Python runtime
FROM python:3.11-slim

# Set the maintainer label
LABEL maintainer="sdorbal1@binghamton.edu"

# Set the working directory
WORKDIR /app

# Copy main.py and your trained model
COPY main.py /app/
COPY mnist_newUp_model.h5 /app/

# Copy static folder and its contents
COPY static/ /app/static/
# Copy templates folder and its contents
COPY templates/ /app/templates/

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install required Python libraries
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.13.0 Flask plotly Pillow matplotlib

# Expose port 5000 for the Flask app to listen on
EXPOSE 5000

# Command to run the app
CMD ["python3", "main.py"]
