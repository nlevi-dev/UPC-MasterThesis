mkdir -p data
NEWDATA=$(pwd)/data
cd ../..
OLDDATA=$(pwd)/data
cd data/native/preprocessed
for d in * ; do
        mkdir -p "$NEWDATA/native/preprocessed/$d"
        cp "$OLDDATA/native/preprocessed/$d/t1_radiomics_raw_k5_b25.npy" "$NEWDATA/native/preprocessed/$d/t1_radiomics_raw_k5_b25.npy"
        mkdir -p "$NEWDATA/normalized/preprocessed/$d"
done
cd "$NEWDATA/.."
zip -r data.zip data
mv data.zip client
rm -rf data