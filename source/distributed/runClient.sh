docker stop levi-master-thesis-client
docker rm levi-master-thesis-client
mkdir -p levi-master-thesis-client
docker run -itd --name levi-master-thesis-client \
    -v "$(pwd)/levi-master-thesis-client:/home/python/mount" \
    -e NAME=UNDEFINED \
    -e ADDRESS="http://172.17.0.1:3010" \
    -e DEBUG=true \
    -e kernelWidth=5 \
    -e binWidth=25 \
    -e absolute=abs \
    -e inp=t1 \
    -e space=native \
    -e cores=6 \
    levi-master-thesis-client:latest
#docker logs --follow levi-master-thesis-client