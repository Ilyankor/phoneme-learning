cd raw_texts
wget -i list.txt
mv links_list.json ../../prepare_data
rm raw_list.txt
rm list.txt
unzip '*.zip'
rm *.zip
cd ../
rm -r raw_texts/*/