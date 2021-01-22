from datetime import datetime

from _common import set_variable

time = datetime.now().strftime('%Y%m%d%H%M%S')
set_variable('NNI_RELEASE', '999.' + time)
