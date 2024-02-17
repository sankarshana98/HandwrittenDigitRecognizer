## MNIST Digit Prediction Web Application
This project is a web application that predicts handwritten digits using a trained deep learning model. Users can upload an image of a handwritten digit, and the application will provide a prediction along with confidence scores. Additionally, users can provide feedback on the prediction to improve and finetune the model's performance.

## Features
- Upload an image of a handwritten digit for prediction.
- View prediction results including the predicted digit, confidence score, and feature maps.
- Provide feedback on prediction accuracy to improve and finetune the model.

##  Technologies Used
Python: Backend language for model training and web server.

Flask: Python web framework used for building the web application.

TensorFlow/Keras: Deep learning library used for training and loading the digit recognition model.

Plotly: JavaScript graphing library used for displaying prediction results.

Pillow: Python Imaging Library used for image processing.

HTML/CSS: Frontend for user interface and styling.

Docker: Containerization tool used for packaging the application.

## Deployment to Google Cloud Run

Google Cloud Run allows you to run stateless containers on a fully managed serverless platform. It automatically scales up or down to handle your application's traffic, and you only pay for the resources you use.

### Prerequisites
Before deploying the application to Google Cloud Run, ensure you have the following prerequisites:

Google Cloud Platform (GCP) Account: You need a GCP account to use Google Cloud Run. If you don't have one, you can sign up [here](https://cloud.google.com/?hl=en).

Google Cloud SDK: Install the Google Cloud SDK on your local machine. You can download it from [here](https://cloud.google.com/sdk/docs/install-sdk).

Docker: Ensure you have Docker installed on your system as Cloud Run requires containerized applications.

### Build the Docker Image:

Build the Docker image for your application using the provided Dockerfile:

                          docker build -t gcr.io/[PROJECT_ID]/mnist-digit-prediction .
                          
Replace [PROJECT_ID] with your GCP project ID.

### Push Docker Image to Container Registry:
Push the Docker image to Google Container Registry (GCR) to store your container image:

                          docker push gcr.io/[PROJECT_ID]/mnist-digit-prediction

### Access the Deployed Application:
After the deployment is complete, you will receive a URL where your application is hosted. You can access the deployed application by navigating to this URL in your web browser.

## Continuous Deployment with Cloud Build
To enable continuous deployment with Google Cloud Build, you can set up a trigger that automatically builds and deploys your application whenever changes are pushed to your repository. Here's how you can set up a Cloud Build trigger:

### Create a Cloud Build Trigger:
Go to the Cloud Build page in the Google Cloud Console and click on "Triggers". Click "Create Trigger" and configure the trigger to monitor your repository for changes.

### Specify Build Configuration:
Create a cloudbuild.yaml file in your repository to define the build steps. This file should include steps for building the Docker image, pushing it to Container Registry, and deploying it to Cloud Run.

### Connect Cloud Build to Google Cloud Run:
Grant the necessary permissions to Cloud Build to deploy to Cloud Run by assigning the Cloud Run Admin role to the Cloud Build service account.

With these steps, Cloud Build will automatically build and deploy your application whenever changes are pushed to your repository, providing a seamless deployment process.

## Contributions
We welcome contributions from the community to improve this project! If you'd like to contribute, follow these simple steps:
- Fork the Repository: Click the "Fork" button at the top-right corner of the repository to create your copy.
- Clone the Repository: Clone your forked repository to your local machine using Git:
  
                           git clone https://github.com/your-username/mnist-GCP-Docker.git
  
- Make Changes: Make your desired changes to the codebase.
- Test: Test your changes locally to ensure they work as expected.
- Commit Changes: Commit your changes to your forked repository:
  
                          git add .
                          git commit -m "Your descriptive commit message"
  
- Push Changes: Push your changes to your fork on GitHub:
  
                          git push origin master
  
- Create Pull Request: Go to your forked repository on GitHub and click the "New pull request" button. Provide a descriptive title and detailed description for your pull request.
