# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time
import traceback
from utils import GREEN, RED, CLEAR, setup_experiment

def test_nni_cli():
    import nnicli as nc

    config_file = 'config_test/examples/mnist-tfv1.test.yml'

    try:
        # Sleep here to make sure previous stopped exp has enough time to exit to avoid port conflict
        time.sleep(6)
        print(GREEN + 'Testing nnicli:' + config_file + CLEAR)
        nc.start_nni(config_file)
        time.sleep(3)
        nc.set_endpoint('http://localhost:8080')
        print(nc.version())
        print(nc.get_job_statistics())
        print(nc.get_experiment_status())
        nc.list_trial_jobs()

        print(GREEN + 'Test nnicli {}: TEST PASS'.format(config_file) + CLEAR)
    except Exception as error:
        print(RED + 'Test nnicli {}: TEST FAIL'.format(config_file) + CLEAR)
        print('%r' % error)
        traceback.print_exc()
        raise error
    finally:
        nc.stop_nni()

if __name__ == '__main__':
    installed = (sys.argv[-1] != '--preinstall')
    setup_experiment(installed)

    test_nni_cli()
