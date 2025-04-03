# Robust Stock Backtesting & ML Trading Tool

This tool ingests and enriches historical stock data, backtests trading strategies with realistic execution, and uses an LSTM model for price forecasting. It provides an interactive Streamlit UI for users to trigger these actions.

## Quick Start

### Prerequisites
- **Docker** installed on your machine
- A **.env** file with your API keys and credentials (see below)

### 1. Create a `.env` File
In your project root, create a file named `.env` (make sure it's added to your `.gitignore`):

```
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY
NEWS_API_KEY=YOUR_NEWS_API_KEY
BOOTSTRAP_SERVERS=YOUR_KAFKA_BOOTSTRAP_SERVER
SASL_USERNAME=YOUR_KAFKA_USERNAME
SASL_PASSWORD=YOUR_KAFKA_PASSWORD
PORT=8080
```

### 2. Build the Docker Image
Open a terminal in your project root (where the Dockerfile is located) and run:

```bash
docker buildx build --platform linux/amd64 -t backtest-ml-app:latest --load .
```

This command builds a Linux (amd64) image and loads it into your local Docker.

### 3. Run the Docker Container Locally
Run the container while loading your environment variables:

```bash
docker run -p 8501:8501 --env-file .env --name backtest-ml-container backtest-ml-app:latest
```

The container maps port 8501 (Streamlit's default) so you can access the app at http://localhost:8501.

### 4. Verify the Application
- Open http://localhost:8501 in your web browser
- Use the UI buttons to fetch data, train the ML model, and run the backtest
- Confirm that the app loads correctly and shows the expected charts and metrics

### 5. Push the Image to a Registry (Optional for Cloud Deployment)
Once you're happy with the local testing, push the image to a container registry:

Tag your image:
```bash
docker tag backtest-ml-app:latest yourdockerusername/backtest-ml-app:latest
```

Push to Docker Hub:
```bash
docker push yourdockerusername/backtest-ml-app:latest
```

### 6. Deploy to the Cloud (Using Google Cloud Run)
If you choose to deploy:

- In Google Cloud Run, create a new service using the image `docker.io/yourdockerusername/backtest-ml-app:latest`
- Ensure the container listens on the port specified by the PORT variable (default 8080)
- Set the environment variables in the Cloud Run configuration (or via a CI/CD pipeline)

## Summary
1. Create your `.env` file with your secrets
2. Build the Docker image with `docker buildx build --platform linux/amd64 -t backtest-ml-app:latest --load .`
3. Run it locally with `docker run -p 8501:8501 --env-file .env --name backtest-ml-container backtest-ml-app:latest`
4. Verify at http://localhost:8501
5. (Optional) Tag and push the image for cloud deployment
6. Deploy to Google Cloud Run by setting up a service with the pushed image and required environment variables

This flow will help you confirm the app works locally and set you up for later cloud deployment.
