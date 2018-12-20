 #!/bin/bash
 echo "Copy nni sdk to remote machine..."
 scp -i $key -r ../src/sdk/pynni $ip:~
 echo "Install nni sdk in remote machine..."
 ssh -i $key $ip "cd pynni && python3 -m pip install --user ."