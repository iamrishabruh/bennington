# Dockerfile

# Use Python 3.10 slim image (Linux-based)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Expose the default port for Streamlit
EXPOSE 8501

# Set the default command to run your Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
