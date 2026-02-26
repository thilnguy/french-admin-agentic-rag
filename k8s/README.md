# Kubernetes Deployment Guide

This directory contains the Kubernetes manifests required to deploy the French Administrative Agentic RAG system in a production environment.

## Architecture

The deployment consists of:
- **rag-api**: The stateless FastAPI application (scalable via HPA).
- **redis**: In-memory data store for agent session state and caching.
- **qdrant**: Stateful vector database using persistent volume claims to store document embeddings.
- **ingress**: NGINX Ingress configuration to expose the API to the Internet, tailored for Server-Sent Events (SSE) streaming capabilities.
- **configmap & secrets**: Centralized configuration management.

## Prerequisites

- A running Kubernetes cluster (e.g., Minikube, EKS, GKE).
- `kubectl` configured to communicate with the cluster.
- An Ingress controller (e.g., NGINX Ingress Controller).
- *Optional*: A Secret Management tool like SealedSecrets or HashiCorp Vault.

## Deployment Steps

1. **Create Namespace**
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. **Configure Secrets**
   Edit `secrets.enc.yaml` to include your actual `OPENAI_API_KEY` and other sensitive variables.
   ```bash
   kubectl apply -f secrets.enc.yaml
   ```

3. **Deploy Configurations and Data Stores**
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f deployment-qdrant.yaml
   kubectl apply -f deployment-redis.yaml
   ```

4. **Deploy the Stateless API**
   Ensure you have built and pushed the Docker image for the API to a container registry accessible by your cluster. Update `deployment-api.yaml` with the correct `image` tag.
   ```bash
   kubectl apply -f deployment-api.yaml
   ```

5. **Apply Autoscaling and Networking**
   Update `ingress.yaml` to match your domain name.
   ```bash
   kubectl apply -f hpa.yaml
   kubectl apply -f ingress.yaml
   ```

## Managing Vector Data
Since Qdrant is stateful, data persists across pod restarts thanks to the `qdrant-pvc` PersistentVolumeClaim. When migrating or restoring, you can use the Qdrant Snapshot API.
