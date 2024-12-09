rm -rf source
rm source.zip
mkdir -p source/data/models
NEWDATA=$(pwd)/source/data
OLDDATA=$(pwd)/data
cp -r data/preprocessed "$NEWDATA"
cd data/native/preloaded
for d in * ; do
    l=$(expr length $d)
    if [ "5" -eq $l ]
    then
        mkdir -p "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k5_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k7_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k9_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k11_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k13_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k15_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k17_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k19_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_left_k21_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/connectivity_left.npy" "$NEWDATA/native/preloaded/$d"
    fi
done
cd ../../..
cp main_feature_selection_client.py "$NEWDATA/.."
cp main_feature_selection_server.py "$NEWDATA/.."
cp util.py "$NEWDATA/.."
cp LayeredArray.py "$NEWDATA/.."
cp ModelClassificationFNN.py "$NEWDATA/.."
cp DataGeneratorClassificationFNN.py "$NEWDATA/.."
zip -r source.zip source
rm -rf source