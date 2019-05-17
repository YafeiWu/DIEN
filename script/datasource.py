import tensorflow as tf

epsilon = 0.000000001
EPR_THRESHOLD = 7.0

def base64_to_int32(base64string):
    decoded = tf.decode_base64(base64string)
    record = tf.decode_raw(decoded, tf.int32)
    return record


def get_one_group(feats_batches, f):
        print("DEBUG INFO -> f_name :{}, f_width: {}, f_seqsize:{}, f_offset:{}, f_ends:{}".format(f.f_name, f.f_width, f.f_seqsize, f.f_offset, f.f_ends))
        feat = feats_batches[:, f.f_offset: f.f_ends] if f.f_width * f.f_seqsize >1 else feats_batches[:, f.f_offset]
        if f.f_type != tf.int32:
            return tf.cast(feat, dtype=f.f_type)
        else:
            return feat

def expand_label(weight):
    target_ = tf.expand_dims(weight, -1) - tf.constant(EPR_THRESHOLD, dtype=tf.float32)
    label = tf.concat([target_, -target_], -1)
    ones_ = tf.ones_like(label)
    zeros_ = tf.zeros_like(label)
    target_ph_ = tf.where(label>0.0, x=ones_, y=zeros_)
    return target_ph_


