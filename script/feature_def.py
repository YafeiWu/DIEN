import json
import logging
logging.basicConfig(level=logging.INFO)

class UserSeqFeature:
    def __init__(self,fname="",foffset=0,fends=0,fwidth=1,fdepth=0,key_num=1,key_set=None,bounds=None,defaultVal=0,group=""):
        self.f_name=fname
        self.f_offset=foffset
        self.f_ends=fends
        self.f_width=fwidth #how many positions in the final feature vector
        # self.f_depth=fdepth #scala vs vector
        # self.key_set = key_set
        # self.bounds = bounds
        # self.key_num = key_num #used for tensorflow variable declaration
        # self.feat_type = "embed or scala" #hash_str
        # self.defaultVal = defaultVal
        # self.group = group

    def __str__(self):
        return "UserSeqFeature({})".format(json.dumps({'f_name': self.f_name, 'f_offset': self.f_offset, 'f_ends': self.f_ends, 'f_width': self.f_width}))

    def copy(self,another_feat):
        self.f_name=another_feat.f_name
        self.f_offset=another_feat.f_offset
        self.f_ends=another_feat.f_ends
        self.f_width=another_feat.f_width #how many positions in the final feature vector
        # self.f_depth=another_feat.f_depth #scala vs vector
        # self.key_set = another_feat.key_set
        # self.bounds = another_feat.bounds
        # self.key_num = another_feat.key_num #used for tensorflow variable declaration
        # self.feat_type = another_feat.feat_type
        # self.defaultVal = another_feat.defaultVal
        # self.group = another_feat.group
