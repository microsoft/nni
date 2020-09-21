FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY mnist.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

###########################################################
RUN pip install nni==1.8.16+petuum --extra-index-url http://pypi.int.petuum.com:8080/simple --trusted-host pypi.int.petuum.com
# Here <YOUR_NNI_VERSION> should matches your local
#   `nnictl --version` just installed above.
#
# Note that for now the experiment codes are allowed
#   to not use any NNI SDK library, but as for the
#   DockerFile its pip dependency is required.
###########################################################
