docker stop levi-master-thesis-server
docker rm levi-master-thesis-server
mkdir -p levi-master-thesis-server
docker run -itd --name levi-master-thesis-server \
    -v "$(pwd)/levi-master-thesis-server:/home/python/mount" \
    --restart always \
    -p 3010:3001 \
    levi-master-thesis-server:latest
docker logs --follow levi-master-thesis-server