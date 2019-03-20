import tensorflow as tf

class Controller(object):
  def __init__(self, *args, **kwargs):
    raise NotImplementedError("Abstract method.")

  def _build_sample(self):
    raise NotImplementedError("Abstract method.")

  def _build_greedy(self):
    raise NotImplementedError("Abstract method.")

  def _build_trainer(self):
    raise NotImplementedError("Abstract method.")
