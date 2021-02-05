在 Google Colab 上使用 NNI
===============================

在 Google Colab 上轻松使用 NNI。 Colab 没有暴露它的公网 IP 及端口，因此默认情况下无法在 Colab 中访问 NNI 的 Web 界面。 为解决此问题，需要使用反向代理软件，例如 ``ngrok`` 或 ``frp``。 此教程将展示如何使用 ngrok 在 Colab 上访问 NNI 的Web 界面。

如何在 Google Colab 上打开 NNI 的 Web 界面
--------------------------------------------------


#. 安装需要的包和软件。

.. code-block:: bash

   ! pip install nni # install nni
   ! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip # download ngrok and unzip it
   ! unzip ngrok-stable-linux-amd64.zip
   ! mkdir -p nni_repo
   ! git clone https://github.com/microsoft/nni.git nni_repo/nni # clone NNI's offical repo to get examples


#. 在 `here <https://ngrok.com/>`__ 注册 ngrok 账号，然后通过 authtoken 来连接。

.. code-block:: bash

   ! ./ngrok authtoken <your-authtoken>


#. 在大于 1024 的端口号上启动 NNI 样例，之后在相同端口上启动 ngrok。 如果希望使用 GPU，确保 config.yml 中 gpuNum >= 1 。 因为使用 ``ngrok http 5000 &`` 会停止响应，所以要使用 ``get_ipython()`` 来启动 ngrok。

.. code-block:: bash

   ! nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 &
   get_ipython().system_raw('./ngrok http 5000 &')


#. 查看公网 url 。

.. code-block:: bash

   ! curl -s http://localhost:4040/api/tunnels # don't change the port number 4040

在步骤 4 后将会看到类似 http://xxxx.ngrok.io 的 url，打开此url即可看到 NNI 的Web 界面。 玩得开心 :)

使用 frp 访问 Web 界面
----------------------

frp 是另一款提供了相似功能的反向代理软件。 但 frp 不提供免费的公网 url，因此可能需要一台拥有公网 IP 的服务器来作为 frp 的服务器端。 参考 `这里 <https://github.com/fatedier/frp>`__ 来了解如何部署 frp。
