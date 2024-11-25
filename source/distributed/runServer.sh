docker build --tag cricks-backend .
docker stop cricks-backend
docker rm cricks-backend
docker save -o cricks-backend.tar cricks-backend
cp cricks-backend.tar /home/levente/nfs/cricks-backend.tar
#docker run -itd --name cricks-backend \
#    --ipc=host \
#    --restart always \
#    -p 3001:3001 \
#    -v /home/levente/nfs/public/materials/models/cricks/raw:/app/files \
#    cricks-backend:latest
