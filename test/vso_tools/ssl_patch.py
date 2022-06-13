"""Fix the issue: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129).

I think it's caused by some incorrect certificates injected by WinRM,
but I guess disabling the check is a simpler way.
"""

import ssl
from pathlib import Path

ssl_file_path = ssl.__file__
print('SSL file path:', ssl_file_path)

# https://stackoverflow.com/questions/36600583/python-3-urllib-ignore-ssl-certificate-verification
old_line = '_create_default_https_context = create_default_context'
new_line = '_create_default_https_context = _create_unverified_context'

new_ssl_file_content = Path(ssl_file_path).read_text().replace(old_line, new_line)
Path(ssl_file_path).write_text(new_ssl_file_content)
