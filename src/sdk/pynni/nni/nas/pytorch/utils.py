from collections import OrderedDict

_counter = 0


def global_mutable_counting():
    global _counter
    _counter += 1
    return _counter


class AverageMeterGroup:

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __str__(self):
        return "  ".join(str(v) for _, v in self.meters.items())


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class StructuredMutableTreeNode:
    def __init__(self, mutable):
        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):
        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):
        return type(self.mutable)

    def traverse(self, order="pre", deduplicate=True, memo=None):
        if memo is None:
            memo = set()
        assert order in ["pre", "post"]
        if order == "pre":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
        for child in self.children:
            for m in child.traverse(order=order, deduplicate=deduplicate, memo=memo):
                yield m
        if order == "post":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
