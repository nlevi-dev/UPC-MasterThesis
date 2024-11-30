cd $1
for d in * ; do
    if [ "${d: -6}" == ".ipynb" ]
    then
        jupyter nbconvert --execute --to notebook --inplace $d
    fi
done