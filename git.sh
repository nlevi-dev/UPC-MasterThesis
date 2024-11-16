sudo chmod 777 -R report
sudo chown levente:levente -R report
sudo chmod 777 -R experiments
sudo chown levente:levente -R experiments
sudo chmod 777 -R source/data/native/models
sudo chown levente:levente -R source/data/native/models
sudo chmod 777 -R source/data/normalized/models
sudo chown levente:levente -R source/data/normalized/models
git add -A
git commit -m "$1"
git push
