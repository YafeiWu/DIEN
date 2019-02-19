#export PATH="~/anaconda4/bin:$PATH"
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
#gunzip reviews_Books.json.gz
#gunzip meta_Books.json.gz
python process_data.py
python local_aggretor.py
python split_by_user.py
python generate_voc.py
echo "ALL DONE!"
