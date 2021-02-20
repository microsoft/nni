主要逻辑在tuner.py + trail.py：相当于就是把arch2vec repo里的dngo_darts.py“翻译”成nni的tuner+trial。
tuner里的generate_paramaters先生成第一批arch（有16个,一个一个发给trial（query）)，receive_trial_result收完一批arch的result以后在generate_paramaters更新DNGO model然后生成新的一批arch(每次5个)，如此循环。

embedding_file => init 16 arch
16 arch => trial => DNGO model => new 5 arch
        => trial => DNGO model => new 5 arch ...
 

how to run:

1、根据https://github.com/MSU-MLSys-Lab/arch2vec 所说的“arch2vec extraction
bash run_scripts/extract_arch2vec_darts.sh
The extracted arch2vec will be saved in ./pretrained/dim-16/arch2vec-darts.pt.
Alternatively, you can download the pretrained arch2vec on DARTS search space.”,

下载embedding file：https://drive.google.com/file/d/1bDZCD-XDzded6SRjDUpRV6xTINpwTNcm/view

2、config.yml 里修改tuner的codeDir和embedding_path

3、nnictl create --config nni/examples/trials/dngo_darts/config.yml

 

没有直接把embedding file用git上传到我的forked nni repo的原因：

embedding file超过100M所以Github不让上传，得用Git LFS 上传大文件，但是
You can't use Git LFS on a fork. Git LFS on github.com does not currently support pushing LFS objects to public forks. 

 

=======================================

替代方案(肯定能跑通)，直接

ssh v-ayanmao@SRGWS-10
conda activate arch2vec_env
nnictl create --config /home/v-ayanmao/msra/nni/examples/trials/may/config.yml