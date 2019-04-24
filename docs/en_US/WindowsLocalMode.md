# Windows Local Mode (experimental feature)
Up to now we only support local mode on Windows. Windows 10.1809 is well tested and recommended.

## **Installation on Windows**

  **Anaconda python(64-bit) is highly recommended**  
  If you use official python and pip install nni, make sure you have one of `Visual Studio`, `MATLAB`, `MKL` and `Intel Distribution for Python` installed on Windows before running nni.  
  If not, scipy install problem missing LIBIFCOREMD.DLL and LIBMMD.DLL will happen, you'd better install one of the softwares above to solve it or change to use Anaconda python(64-bit).
  
When you use powershell to run script for the first time, you need run powershell as Administrator with this command:
```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

* __Install NNI through pip__

  Prerequisite: `python(64-bit) >= 3.5`
  ```bash
  python -m pip install --upgrade nni
  ```

* __Install NNI through source code__

  Prerequisite: `python >=3.5`, `git`, `powershell`
  ```bash
  git clone -b v0.7 https://github.com/Microsoft/nni.git
  cd nni
  powershell ./install.ps1
  ```
Note
  * install simple json package?

When these things are done, run the **config_windows.yml** file from your command line to start the experiment.

```bash
    nnictl create --config nni/examples/trials/mnist/config_windows.yml
```
For other examples you need to change trial command `python3` into `python` in each example yaml.
  
