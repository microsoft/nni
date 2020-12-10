# Neural Architecture Search Comparison
*Posted by Anonymous Author*

Train and Compare NAS (Neural Architecture Search) models including Autokeras, DARTS, ENAS and NAO.

Their source code link is as below:

- Autokeras: [https://github.com/jhfjhfj1/autokeras](https://github.com/jhfjhfj1/autokeras)

- DARTS: [https://github.com/quark0/darts](https://github.com/quark0/darts)

- ENAS: [https://github.com/melodyguan/enas](https://github.com/melodyguan/enas)

- NAO: [https://github.com/renqianluo/NAO](https://github.com/renqianluo/NAO)

## Experiment Description

To avoid over-fitting in **CIFAR-10**, we also compare the models in the other five datasets including Fashion-MNIST, CIFAR-100, OUI-Adience-Age, ImageNet-10-1 (subset of ImageNet), ImageNet-10-2 (another subset of ImageNet). We just sample a subset with 10 different labels from ImageNet to make ImageNet-10-1 or ImageNet-10-2.

| Dataset                                                      | Training Size | Numer of Classes | Descriptions                                                 |
| :----------------------------------------------------------- | ------------- | ---------------- | ------------------------------------------------------------ |
| [Fashion-MNIST](<https://github.com/zalandoresearch/fashion-mnist>) | 60,000        | 10               | T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. |
| [CIFAR-10](<https://www.cs.toronto.edu/~kriz/cifar.html>)    | 50,000        | 10               | Airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships and trucks. |
| [CIFAR-100](<https://www.cs.toronto.edu/~kriz/cifar.html>)   | 50,000        | 100              | Similar to CIFAR-10 but with 100 classes and 600 images each. |
| [OUI-Adience-Age](<https://talhassner.github.io/home/projects/Adience/Adience-data.html>) | 26,580        | 8                | 8 age groups/labels (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-). |
| [ImageNet-10-1](<http://www.image-net.org/>)                 | 9,750         | 10               | Coffee mug, computer keyboard, dining table, wardrobe, lawn mower, microphone, swing, sewing machine, odometer and gas pump. |
| [ImageNet-10-2](<http://www.image-net.org/>)                 | 9,750         | 10               | Drum, banj, whistle, grand piano, violin, organ, acoustic guitar, trombone, flute and sax. |



We do not change the default fine-tuning technique in their source code. In order to match each task, the codes of input image shape and output numbers are changed.

Search phase time for all NAS methods is **two days** as well as the retrain time.  Average results are reported based on **three repeat times**. Our evaluation machines have one Nvidia Tesla P100 GPU, 112GB of RAM and one 2.60GHz CPU (Intel E5-2690).

For NAO, it requires too much computing resources, so we only use NAO-WS which provides the pipeline script.

For AutoKeras, we used  0.2.18 version because it was the latest version when we started the experiment.

## NAS Performance

| NAS             | AutoKeras (%) | ENAS (macro) (%) | ENAS (micro) (%) | DARTS (%) | NAO-WS (%) |
| --------------- | :-----------: | :--------------: | :--------------: | :-------: | :--------: |
| Fashion-MNIST   |     91.84     |      95.44       |      95.53       | **95.74** |   95.20    |
| CIFAR-10        |     75.78     |      95.68       |    **96.16**     |   94.23   |   95.64    |
| CIFAR-100       |     43.61     |      78.13       |      78.84       | **79.74** |   75.75    |
| OUI-Adience-Age |     63.20     |    **80.34**     |      78.55       |   76.83   |   72.96    |
| ImageNet-10-1   |     61.80     |      77.07       |      79.80       | **80.48** |   77.20    |
| ImageNet-10-2   |     37.20     |      58.13       |      56.47       |   60.53   | **61.20**  |

Unfortunately, we cannot reproduce all the results in the paper.

The best or average results reported in the paper:

| NAS       | AutoKeras(%) | ENAS (macro) (%) | ENAS (micro) (%) |   DARTS (%)    | NAO-WS (%)  |
| --------- | ------------ | :--------------: | :--------------: | :------------: | :---------: |
| CIFAR- 10 | 88.56(best)  |   96.13(best)    |   97.11(best)    | 97.17(average) | 96.47(best) |

For AutoKeras, it has relatively worse performance across all datasets due to its random factor on network morphism.

For ENAS, ENAS (macro) shows good results in OUI-Adience-Age and ENAS (micro)  shows good results in CIFAR-10.

For DARTS, it has a good performance on some datasets but we found its high variance in other datasets. The difference among three runs of benchmarks can be up to 5.37% in OUI-Adience-Age and 4.36% in ImageNet-10-1.

For NAO-WS, it shows good results in ImageNet-10-2 but it can perform very poorly in OUI-Adience-Age.

## Reference

1. Jin, Haifeng, Qingquan Song, and Xia Hu. "Efficient neural architecture search with network morphism." *arXiv preprint arXiv:1806.10282* (2018).

2. Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).

3. Pham, Hieu, et al. "Efficient Neural Architecture Search via Parameters Sharing." international conference on machine learning (2018): 4092-4101.

4. Luo, Renqian, et al. "Neural Architecture Optimization." neural information processing systems (2018): 7827-7838.
