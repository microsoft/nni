# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import time
import traceback
from utils import GREEN, RED, CLEAR, setup_experiment

def test_nni_cli():
    import nnicli as nc

    config_file = 'config_test/examples/mnist.test.yml'

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
