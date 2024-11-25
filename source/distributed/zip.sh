mkdir -p data
NEWDATA=$(pwd)/data
cd ..
OLDDATA=$(pwd)/data
cp -r data/preprocessed "$NEWDATA"
cd data/raw
for d in * ; do
  l=$(expr length $d)
  if [ "5" -eq $l ]
  then
    mkdir -p "$NEWDATA/native/preprocessed/$d"
    cp "$OLDDATA/native/preprocessed/$d/t1.npy" "$NEWDATA/native/preprocessed/$d/t1.npy" 2>/dev/null
    cp "$OLDDATA/native/preprocessed/$d/t1t2.npy" "$NEWDATA/native/preprocessed/$d/t1t2.npy" 2>/dev/null
    cp "$OLDDATA/native/preprocessed/$d/mask_brain.npy" "$NEWDATA/native/preprocessed/$d/mask_brain.npy" 2>/dev/null
    mkdir -p "$NEWDATA/normalized/preprocessed/$d"
    cp "$OLDDATA/normalized/preprocessed/$d/t1.npy" "$NEWDATA/normalized/preprocessed/$d/t1.npy" 2>/dev/null
    cp "$OLDDATA/normalized/preprocessed/$d/t1t2.npy" "$NEWDATA/normalized/preprocessed/$d/t1t2.npy" 2>/dev/null
    cp "$OLDDATA/normalized/preprocessed/$d/mask_brain.npy" "$NEWDATA/normalized/preprocessed/$d/mask_brain.npy" 2>/dev/null
  fi
done
cd "$NEWDATA/.."
rm -f client/data.zip
zip -r data.zip data
mv data.zip client
rm -rf data