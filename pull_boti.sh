mv source/data/models/hashes.txt source/data/models/hashes.bak
git checkout .
git pull
rm source/data/models/hashes.txt
mv source/data/models/hashes.bak source/data/models/hashes.txt
