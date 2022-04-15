The Experiments in figure 16 are on Nvidia A100 GPUs. To reporduce the figure16, we need first build the docker image that contains all the dependencies we need. The docker image is at `$SPARTA_HOME/script/Dockerfile.a100`. After building the image successfully, we can run the experiment by the following commands.

```
cd figure16/a100
bash run.sh
```

The data will be generated and visialized automatically and the result PDF will be saved at `figure16/a100`.