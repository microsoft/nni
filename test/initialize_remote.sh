 #!/bin/bash
 ip="nni@104.210.63.241"
 key="id_rsa"
 chmod 600 $key
 echo "Initializing remote machine..."
 yes | ssh -i $key $ip "rm -rf pynni"
 echo "Copy nni sdk to remote machine..."
 scp -i $key -r ../src/sdk/pynni $ip:~
 echo "Install nni sdk in remote machine..."
 ssh -i $key $ip "cd pynni && python3 -m pip install --user ."