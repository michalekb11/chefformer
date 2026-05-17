import base64

def get_base64_of_bin_file(bin_file):
    """
    Encodes a local binary file (like an image) to a base64 string.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()