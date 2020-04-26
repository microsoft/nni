**设置 NNI 开发环境**

===

## 调试 NNI 源代码的最佳实践

要调试 NNI 源代码，需要 Ubuntu 16.04 或更高版本系统的开发环境，并需要安装 Python 3 以及 pip 3，然后遵循以下步骤。

### 1. Clone the source code

Run the command

    git clone https://github.com/Microsoft/nni.git
    

to clone the source code

### 2. Prepare the debug environment and install dependencies**

Change directory to the source code folder, then run the command

    make install-dependencies
    

to install the dependent tools for the environment

### 3. Build source code

Run the command

    make build
    

to build the source code

### 4. Install NNI to development environment

Run the command

    make dev-install
    

to install the distribution content to development environment, and create cli scripts

### 5. Check if the environment is ready

Now, you can try to start an experiment to check if your environment is ready. For example, run the command

    nnictl create --config ~/nni/examples/trials/mnist-tfv1/config.yml
    

And open WebUI to check if everything is OK

### 6. Redeploy

After the code changes, it may need to redeploy. It depends on what kind of code changed.

#### Python

It doesn't need to redeploy, but the nnictl may need to be restarted.

#### TypeScript

* If `src/nni_manager` will be changed, run `yarn watch` continually under this folder. It will rebuild code instantly.
* If `src/webui` or `src/nasui` is changed, use **step 3** to rebuild code.

The nnictl may need to be restarted.

* * *

At last, wish you have a wonderful day. For more contribution guidelines on making PR's or issues to NNI source code, you can refer to our [Contributing](Contributing.md) document.