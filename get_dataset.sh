TRAJ_URL=https://seafile.lirmm.fr/seafhttp/files/a5bb8933-0bc8-4e21-bb7f-31d802ec83eb/archaeo_sequence_6_raw_data.tar.gz


cd
mkdir -p aqualoc_dataset/images_sequence
cd aqualoc_dataset
sudo wget $TRAJ_URL
tar -xf archaeo_sequence_6_raw_data.tar.gz --wildcards '*.png'
mv raw_data/images_sequence_6/*.png ./images_sequence
rm -r raw_data