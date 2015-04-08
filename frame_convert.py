from numpy import clip, uint8


def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array that has been processed whos datatype is unspecified
    """
    clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(uint8)
    return depth
