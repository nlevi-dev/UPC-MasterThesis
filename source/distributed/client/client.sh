if [ ! -f "mount/data.zip" ]; then
    #setup data
    if [ ! -d "mount/data" ]; then
        unzip data.zip
        mv data mount/data
    fi

    #computation
    python client.py

    #delete unused
    cd mount/data
    rm -rf preprocessed
    cd native/preprocessed
    for d in * ; do
        rm "$d/t1.npy" 2>/dev/null
        rm "$d/t1t2.npy" 2>/dev/null
        rm "$d/mask_brain.npy" 2>/dev/null
    done
    cd ../../normalized/preprocessed
    for d in * ; do
        rm "$d/t1.npy" 2>/dev/null
        rm "$d/t1t2.npy" 2>/dev/null
        rm "$d/mask_brain.npy" 2>/dev/null
    done
    cd ../../..

    #zip result
    zip -r data.zip data
    rm -rf data
    cd ..
fi

#upload data
while true
do
    curl -X POST -F result=@mount/data.zip "$ADDRESS/upload/$NAME" && break || sleep 600
done

#remove zip and log
rm mount/data.zip
