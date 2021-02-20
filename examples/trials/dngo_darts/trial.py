import logging
import numpy as np
import torch

import nni

from dngo_darts import query

_logger = logging.getLogger('May_DNGO')
_logger.setLevel(logging.INFO)





def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset 
      - Train the model
      - Report accuracy to tuner
    """
    # val_acc, test_acc = query(count, args.seed, params, args.inner_epochs)
    val_acc, test_acc = query(981230, 3, params, 4)
    _logger.info('Final accuracy reported: %s, %s', val_acc, test_acc)

    res = {"default":val_acc, "test_acc":test_acc}
    # send final accuracy to NNI tuner and web UI
    nni.report_final_result(res)


if __name__ == '__main__':

    tuned_params = nni.get_next_parameter()
    main(tuned_params)

