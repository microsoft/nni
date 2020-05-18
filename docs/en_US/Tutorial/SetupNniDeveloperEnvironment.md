**Set up NNI developer environment**

===

## Best practice for debug NNI source code

For debugging NNI source code, your development environment should be under Ubuntu 16.04 (or above) system with python 3 and pip 3 installed, then follow the below steps.

### 1. Clone the source code

Run the command

```
git clone https://github.com/Microsoft/nni.git
```

to clone the source code

### 2. Prepare the debug environment and install dependencies

Change directory to the source code folder, then run the command

```
make install-dependencies
```

to install the dependent tools for the environment

### 3. Build source code

Run the command

```
make build
```

to build the source code

### 4. Install NNI to development environment

Run the command

```
make dev-install
```

to install the distribution content to development environment, and create cli scripts

### 5. Check if the environment is ready

Now, you can try to start an experiment to check if your environment is ready.
For example, run the command

```
nnictl create --config ~/nni/examples/trials/mnist-tfv1/config.yml
```

And open WebUI to check if everything is OK

### 6. Redeploy

After the code changes, it may need to redeploy. It depends on what kind of code changed. 

#### Python

It doesn't need to redeploy, but the nnictl may need to be restarted.

#### TypeScript

* If `src/nni_manager` is changed, run `yarn watch` continually under this folder. It will rebuild code instantly. The nnictl may need to be restarted to reload NNI manager.
* If `src/webui` or `src/nasui` are changed, run `yarn start` under the corresponding folder. The web UI will refresh automatically if code is changed.


---
At last, wish you have a wonderful day.
For more contribution guidelines on making PR's or issues to NNI source code, you can refer to our [Contributing](Contributing.md) document.
