sh zip.sh
cd client

cp ../../RecordHandler.py RecordHandler.py
cp ../../Record.py Record.py
cp ../../extractor_params.py extractor_params.py
cp ../../LayeredArray.py LayeredArray.py
cp ../../util.py util.py
cp ../../visual.py visual.py

docker build --tag levi-master-thesis-client .

rm RecordHandler.py
rm Record.py
rm extractor_params.py
rm LayeredArray.py
rm util.py
rm visual.py

cd ..
rm client/data.zip
docker save -o levi-master-thesis-client.tar levi-master-thesis-client
docker image prune -f