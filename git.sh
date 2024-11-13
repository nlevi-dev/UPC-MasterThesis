sudo chmod 777 *.ipynb
sudo chown levente:levente *.ipynb
sudo chmod 777 -R source/data/models
sudo chown levente:levente -R source/data/models
git add -A
git commit -m "$1"
git push
