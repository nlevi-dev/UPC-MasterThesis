#setup data
if [ ! -d "mount/data" ]; then
    unzip data.zip
    rm -rf mount/data
    mv data mount/data
fi

#computation
python client.py

#upload data
#cd mount
#rm -rf data
#mv data_out data
#zip -r data.zip data
#upload
#&& rm data.zip