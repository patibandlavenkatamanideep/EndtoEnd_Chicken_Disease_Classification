# Chicken-Disease-Classification--Project
This project implements an end-to-end CNN-based deep learning pipeline for classifying chicken diseases. It includes data versioning with DVC, model training, evaluation, Dockerization, and AWS CI/CD deployment using GitHub Actions

## Workflows

1. Update config.yaml - Modify configuration settings as per your environment or requirements.
2. Update secrets.yaml [Optional] - Update any secret keys or credentials if required.
3. Update params.yaml - Change parameters such as hyperparameters, paths, or constants.
4. Update the entity - Modify the data or domain entity class/structure if needed.
5. Update the configuration manager in src/config - Ensure it correctly reads the updated YAML files and provides values to the components.
6. Update the components - Update modules responsible for data ingestion, processing, model training, or other functionalities.
7. Update the pipeline - Make necessary changes to the workflow or orchestrator that runs the components.
8. Update main.py - Ensure the main execution file reflects all new updates and runs the pipeline correctly.
9. Update dvc.yaml - Track updated data or pipeline stages using DVC (Data Version Control).

-------------------------------------------------------------------

## HOW TO RUN THE PROJECT

Clone the repository:
git clone https://github.com/patibandlavenkatamanideep/EndtoEnd_Chicken_Disease_Classification.git
cd Chicken-Disease-Classification--Project

Create a conda environment:
conda create -n cnncls python=3.8 -y
conda activate cnncls

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open the browser and navigate to the localhost and port shown in the terminal.

-------------------------------------------------------------------

## DVC COMMANDS

Initialize DVC:
dvc init

Run the pipeline:
dvc repro

Visualize the pipeline:
dvc dag

-------------------------------------------------------------------

## AWS CI/CD DEPLOYMENT WITH GITHUB ACTIONS

Login to AWS Console.

Create an IAM user with the following access:
- EC2 (Virtual Machine)
- ECR (Elastic Container Registry)

Attach policies:
- AmazonEC2ContainerRegistryFullAccess
- AmazonEC2FullAccess

Create an ECR repository to store Docker images.
Example URI:
566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

Create an EC2 instance:
- OS: Ubuntu

Install Docker on EC2:

sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

Configure EC2 as a self-hosted GitHub runner:
GitHub Repo → Settings → Actions → Runners → New self-hosted runner
Choose OS and execute the commands on EC2.

-------------------------------------------------------------------

## GITHUB SECRETS CONFIGURATION

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME=simple-app

-------------------------------------------------------------------

## DEPLOYMENT FLOW

Build Docker image from source code
Push Docker image to AWS ECR
Launch EC2 instance
Pull Docker image from ECR to EC2
Run Docker container on EC2

-------------------------------------------------------------------

## TECH STACK

Python 3.8
TensorFlow / Keras
CNN
Flask
DVC
Docker
AWS (EC2, ECR)
GitHub Actions

-------------------------------------------------------------------

## AUTHOR

Venkata Manideep Patibandla
