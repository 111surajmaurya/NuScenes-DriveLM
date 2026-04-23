
 sudo docker build -t nuscene .

docker run -it --gpus all \
  -p 8888:8888 \
  -v ~/workspace/assignment:/workspace/assignment \
  -v ~/workspace/assignment/data:/data \
  nuscene


to run jupyter notebook 
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root