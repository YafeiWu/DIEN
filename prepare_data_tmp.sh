set -e
#export PATH="~/anaconda4/bin:$PATH"
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
#gunzip reviews_Books.json.gz
#gunzip meta_Books.json.gz
python script/fix_iteminfo.py
python script/process_data.py
python script/local_aggretor.py
#python script/split_by_user.py
python script/generate_voc.py
echo "ALL DONE! # prepare_data_tmp.sh"
