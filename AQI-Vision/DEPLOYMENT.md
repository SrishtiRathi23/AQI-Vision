# Deployment — Google Cloud Run

Prerequisites: a GCP project, billing enabled, and the `gcloud` CLI installed
and authenticated (`gcloud auth login`).

```bash
# 0. Set variables
export PROJECT_ID=your-gcp-project
export REGION=asia-south1            # Mumbai
export REPO=aqi-vision
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/aqi-vision:latest

# 1. Build the image locally
docker build -t aqi-vision:latest .

# 2. Test locally (open http://localhost:8080)
docker run -p 8080:8080 aqi-vision:latest

# 3. Create an Artifact Registry repo (one-time)
gcloud artifacts repositories create $REPO \
  --repository-format=docker --location=$REGION

# 4. Authenticate Docker to Artifact Registry, tag and push
gcloud auth configure-docker $REGION-docker.pkg.dev
docker tag aqi-vision:latest $IMAGE
docker push $IMAGE

# 5. Deploy to Cloud Run
gcloud run deploy aqi-vision \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080
```

After deploy, `gcloud` prints the public service URL. Paste it into the README
deployment badge and the application form's demo-link field.

> The trained model (`models/best_model.pkl`) and processed features must exist
> in the image. Either run `python train.py` before `docker build`, or run it as
> a build step. Since `data/raw/` is `.dockerignore`d, the container falls back
> to synthetic data unless you bake the processed parquet into the image.
