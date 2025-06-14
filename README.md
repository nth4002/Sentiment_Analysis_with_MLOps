# Run the container ready for training and tracking code on Kaggle notebook

## Build the mlflow image

docker-compose build --no-cache mlflow

## Run the container

docker-compose up -d

## log mlflow console

docker-compose logs mlflow

## check all the containers running

docker ps

## Delete the container

docker-compose down -v
