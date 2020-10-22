

INTERMEDIATE_OFFSET = "intermediate_result_idx_offset"


def sync_intermediate_seq(accum):
    """
    Parameters
    ----------
    accum:
        An Accumulator which persists running state. It is only
        used during AdaptDL training.

    Returns
    -------
    int
        Synchronized intermediate sequence id of current trial job.
    """

    # NOTE: Accumulator should be used under synchronized mode,
    # which can only be given from user script side.
    if not type(accum).__name__.endswith('Accumulator'):
        raise TypeError("'accum' should be an AdaptDL Accumulator.")

    if INTERMEDIATE_OFFSET not in accum:
        accum[INTERMEDIATE_OFFSET] = 0
    intermediate_seq = accum[INTERMEDIATE_OFFSET]
    accum[INTERMEDIATE_OFFSET] += 1
    return intermediate_seq
