class Decision:
    def __init__(self):
        raise NotImplementedError("You should never use init to initialize a general decision.")

    @classmethod
    def from_nni_protocol_format(cls, candidate, search_space):
        assert "_idx" in candidate and "_val" in candidate, "A candidate must have '_idx' and '_val' in its fields."
        assert type(candidate["_idx"]) == type(candidate["_val"]), "Indices and values must have the same type."
        search_space_values = search_space["_values"]
        if isinstance(candidate["_idx"], list):
            assert len(candidate["_idx"]) == len(candidate["_val"]), \
                "Number of indices must be equal of number of values."
            for idx, val in zip(candidate["_idx"], candidate["_val"]):
                assert 0 <= idx < len(search_space_values) and search_space_values[idx] == val, \
                    "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_values, val)
        elif isinstance(candidate["_idx"], int):
            idx, val = candidate["_idx"], candidate["_val"]
            assert 0 <= idx < len(search_space_values) and search_space_values[idx] == val, \
                "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_values, val)
        else:
            raise ValueError("Index of unrecognized type: {}".format(candidate["_idx"]))
        return cls.from_indices(candidate["_idx"], len(search_space_values))

    @classmethod
    def from_indices(cls, indices, n_candidates):
        """
        Construct a decision from indices.

        Parameters
        ----------
        indices : int or list of int
        n_candidates : int

        Returns
        -------
        RelaxedDecision
        """
        return RelaxedDecision(indices, n_candidates)

    @classmethod
    def deserialize(cls, obj):
        pass

    def serialize(self):
        raise NotImplementedError


class RelaxedDecision(Decision):
    def __init__(self, indices, n_candidates):
        if isinstance(indices, int):
            self.indices = [indices]
        elif isinstance(indices, list):
            self.indices = indices
        assert len(set(self.indices)) == len(self.indices), "Indices must be unique"
        assert all(map(lambda x: 0 <= x < n_candidates, self.indices)), "Indices must be in range [0, n_candidates)."
        self.n_candidates = n_candidates

    @classmethod
    def from_multi_hot_iterable(cls, iterable):
        indices, total = [], 0
        for i, t in enumerate(iterable):
            if t:
                indices.append(i)
            total += 1
        return cls(indices, total)
