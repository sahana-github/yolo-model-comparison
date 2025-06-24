# 1. Base image with Python
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy all project files into the container
COPY . /app

# 4. Install required Python packages
RUN pip install --no-cache-dir ultralytics

# 5. Create output folder (if not present)
RUN mkdir -p output

# 6. Set default command to run inference
CMD ["python", "run_inference.py", "--model", "yolov5s.pt", "--input", "test_images", "--output", "output"]
