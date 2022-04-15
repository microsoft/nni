source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
# get the latest nnfusion
pushd /root/nnfusion
git pull origin hubert_antares
pushd build
make -j
popd
popd


HTTP_PORT=8881 BACKEND=c-cuda nohup antares rest-server &
# wait the antares to be ready
sleep 5s
mkdir -p ~/.cache/antares/codehub/
cp figure8/codehub/* ~/.cache/antares/codehub/

# download the checkpoint
azcopy copy "https://nni.blob.core.windows.net/artifact/cks?sp=rl&st=2022-04-13T11:26:02Z&se=2022-12-31T19:26:02Z&sv=2020-08-04&sr=c&sig=rU%2By1QHWTP7jl80p%2FxItK5dVtgvQ6Xpl3rTX%2FNaACX4%3D" "." --recursive
rm -rf checkpoints/bert/checkpoints
rm -rf checkpoints/mobilenet/checkpoints
rm -rf checkpoints/hubert/checkpoints
mv cks/bert/checkpoints checkpoints/bert/
mv cks/mobilenet/checkpoints checkpoints/mobilenet/
mv cks/hubert/checkpoints checkpoints/hubert/ 