import numpy
import json
import cPickle as pkl
import random
import traceback
import gzip

import shuffle

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))

def load_voc(filename):
    try:
        with open(filename, 'rb') as f:
            lines = f.readlines()
            res_dict = {}
            keys = []
            id,index =None,None
            for line in lines:
                arr = json.loads(line.strip())
                if not keys:
                    keys = arr.keys()
                for k in keys:
                    if isinstance(arr[k],(str,unicode)):
                        id = arr[k]
                    elif isinstance(arr[k],(int,long)):
                        index = arr[k]
                    else:
                        print("Type Error # {}".format(type(arr[k])))
                if None not in [id, index]:
                    res_dict[id] = index
                if "DEFAULT" not in res_dict:
                    res_dict['DEFAULT'] = 0
            return res_dict
    except Exception as e:
        print('ERROR {}'.format(traceback.format_exc(e)))


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIteratorV2:

    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_voc(source_dict))

        f_meta = open(item_info, "r")
        meta_map = {}
        self.mid_list_for_random = []
        for line in f_meta:
            arr = json.loads(line.strip())
            id = arr["news_entry_id"]
            cate = arr["category"]
            num = int(arr["eplays"])
            if id not in meta_map:
                meta_map[id] = cate
            tmp_negidx  = self.source_dicts[1][id]
            tmp_negarr = [tmp_negidx]*num
            self.mid_list_for_random.extend(tmp_negarr)

        self.meta_id_map ={}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def print_data_info(self):
        print("source # {}".format(self.source))
        print("n_uid # {}".format(self.n_uid))
        print("n_mid # {}".format(self.n_mid))
        print("n_cat # {}".format(self.n_cat))
        print("mid_list_for_random # {}".format(len(self.mid_list_for_random)))


    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source= shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            print("STOP for end_of_data # {}".format(self.end_of_data))
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))
            print("DEBUG readline source_buffer length #{}".format(len(self.source_buffer)))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            print("STOP for source_buffer # {}".format(len(self.source_buffer)))
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError as e:
                    break
                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                tmp = []
                for fea in ss[4].split(" "):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp
                # if len(mid_list) > self.maxlen:
                #     continue
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                # tmp1 = []
                # for fea in ss[5].split(" "):
                #     c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                #     tmp1.append(c)
                # cat_list = tmp1
                tmp1 = []
                for mid in mid_list:
                    c = self.meta_id_map[mid] if mid in self.meta_id_map else 0
                    tmp1.append(c)
                cat_list = tmp1

                # read from source file and map to word index

                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random)-1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list])
                target.append([float(ss[0]), 1-float(ss[0])])
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
            print("End of data")

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        print("DEBUG data_itertor source length #{}, target length #{}".format(len(source), len(target)))

        return source, target


