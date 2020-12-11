Use NNI on Google Colab
=======================

NNI can easily run on Google Colab platform. However, Colab doesn't expose its public IP and ports, so by default you can not access NNI's Web UI on Colab. To solve this, you need a reverse proxy software like ``ngrok`` or ``frp``. This tutorial will show you how to use ngrok to access NNI's Web UI on Colab.

How to Open NNI's Web UI on Google Colab
----------------------------------------


#. Install required packages and softwares.

.. code-block:: bash

   ! pip install nni # install nni
   ! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip # download ngrok and unzip it
   ! unzip ngrok-stable-linux-amd64.zip
   ! mkdir -p nni_repo
   ! git clone https://github.com/microsoft/nni.git nni_repo/nni # clone NNI's offical repo to get examples


#. Register a ngrok account `here <https://ngrok.com/>`__\ , then connect to your account using your authtoken.

.. code-block:: bash

   ! ./ngrok authtoken <your-authtoken>


#. Start an NNI example on a port bigger than 1024, then start ngrok with the same port. If you want to use gpu, make sure gpuNum >= 1 in config.yml. Use ``get_ipython()`` to start ngrok since it will be stuck if you use ``! ngrok http 5000 &``.

.. code-block:: bash

   ! nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 &
   get_ipython().system_raw('./ngrok http 5000 &')


#. Check the public url.

.. code-block:: bash

   ! curl -s http://localhost:4040/api/tunnels # don't change the port number 4040

You will see an url like http://xxxx.ngrok.io after step 4, open this url and you will find NNI's Web UI. Have fun :)

Access Web UI with frp
----------------------

frp is another reverse proxy software with similar functions. However, frp doesn't provide free public urls, so you may need an server with public IP as a frp server. See `here <https://github.com/fatedier/frp>`__ to know more about how to deploy frp.
