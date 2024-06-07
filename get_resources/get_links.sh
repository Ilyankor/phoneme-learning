wget -w 1 -m -H "https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
cd www.gutenberg.org/robot
cat * >> raw_list.txt
mv raw_list.txt ../../raw_texts
cd ../../
rm -rf aleph.gutenberg.org
rm -rf www.gutenberg.org