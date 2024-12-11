docker stop feature-selection
docker rm feature-selection
docker run -itd --name feature-selection \
    -v "$(pwd):/home/python" \
    --restart always \
    -p 127.0.0.1:15000:15000 \
    feature-selection:latest
docker logs --follow feature-selection