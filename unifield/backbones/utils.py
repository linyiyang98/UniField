import hashlib

def hash_state_dict_keys(state_dict):
    return hashlib.md5(",".join(sorted(state_dict)).encode()).hexdigest()
