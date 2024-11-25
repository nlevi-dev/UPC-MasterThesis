#sh zip.sh
cd client

cp ../../DataHandler.py DataHandler.py
cp ../../DataPoint.py DataPoint.py
cp ../../extractor_params.py extractor_params.py
cp ../../LayeredArray.py LayeredArray.py
cp ../../util.py util.py
cp ../../visual.py visual.py

docker build --tag levi-master-thesis-client .

rm DataHandler.py
rm DataPoint.py
rm extractor_params.py
rm LayeredArray.py
rm util.py
rm visual.py

cd ..
#rm data.zip
#docker save -o levi-master-thesis-client.tar levi-master-thesis-client
#docker image prune -f