docker stop levi-master-thesis-client
docker rm levi-master-thesis-client
mkdir -p levi-master-thesis-client
docker run -itd --name levi-master-thesis-client \
    -v "$(pwd)/levi-master-thesis-client:/home/python/mount" \
    -e NAME=XPS \
    -e ADDRESS="http://172.17.0.1:3010" \
    -e DEBUG=true \
    -e kernelWidth=15 \
    levi-master-thesis-client:latest
docker logs --follow levi-master-thesis-client