# Frontend

## Deploy

```
To remove old dangling docker images:
docker rm $(docker ps -a -f status=exited -q)

Build Docker:
docker build -t medisure-backend .


Check Docker if running correctly:
docker run -e PORT=$PORT -d <image_name>

gcloud builds submit --tag gcr.io/<PROJECT_ID>/<CONTAINER_NAME>
gcloud run deploy --image gcr.io/<PROJECT_ID>/<CONTAINER_NAME> --platform managed
```
