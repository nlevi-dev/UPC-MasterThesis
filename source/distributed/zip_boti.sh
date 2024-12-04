rm -rf data
rm data.zip
mkdir -p data/models
NEWDATA=$(pwd)/data
cd ..
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
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k5_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k7_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k9_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k11_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k13_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k15_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k17_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k19_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/t1_radiomics_norm_right_k21_b25.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/connectivity_left.npy" "$NEWDATA/native/preloaded/$d"
        cp "$OLDDATA/native/preloaded/$d/connectivity_right.npy" "$NEWDATA/native/preloaded/$d"
    fi
done
cd "$NEWDATA/.."
zip -r data.zip data
#rm -rf data