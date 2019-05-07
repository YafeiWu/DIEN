import json
import logging
import tensorflow as tf
logging.basicConfig(level=logging.INFO)

class UserSeqFeature:
    def __init__(self,f_name="", f_offset=0, f_ends=0, f_width=1, f_type=tf.int32, f_embedding=True, f_max=None, f_group='', f_seqsize=1):
        self.f_name = f_name
        self.f_offset = f_offset
        self.f_ends = f_ends
        self.f_width = f_width #how many positions in the final feature vector
        self.f_type = f_type
        self.f_embedding = f_embedding
        self.f_max = f_max
        self.f_group = f_group
        self.f_seqsize = f_seqsize

    def __str__(self):
        return "UserSeqFeature({})".format(json.dumps({'f_name': self.f_name, 'f_offset': self.f_offset,
                                                       'f_ends': self.f_ends, 'f_width': self.f_width,
                                                       'f_embedding': self.f_embedding, 'f_max': self.f_max,
                                                       'f_group':self.f_group, 'f_seqsize': self.f_seqsize
                                                       }))

    def copy(self,another_feat):
        self.f_name = another_feat.f_name
        self.f_offset = another_feat.f_offset
        self.f_ends = another_feat.f_ends
        self.f_width = another_feat.f_width
        self.f_type = another_feat.f_type
        self.f_embedding = another_feat.f_embedding
        self.f_max = another_feat.f_max
        self.f_group = another_feat.f_group
        self.f_seqsize = another_feat.f_seqsize
