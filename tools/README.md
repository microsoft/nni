## NNI CTL
The NNI CTL module is used to control Neural Network Intelligence, including start a new experiment, stop an experiment and update an experiment etc.  
## Environment

```
Ubuntu 16.04 or other Linux OS
python >= 3.5
```

## Installation

 1.Enter tools directory  
 
 2.Use pip3 to install packages

   2.1 Install for current user:

     pip3 install --user -e .

   2.2 Install for all users:

     pip3 install -e .

 
 3.Change the mode of nnictl file 
   
    chmod +x ./nnictl
    
4.Add nnictl to your PATH system environment variable.   
  
  * You could use `export` command to set PATH variable temporary.
  
          export PATH={your nnictl path}:$PATH  

  * Or you could edit your `/etc/profile` file.
          
          1.sudo vim /etc/profile
          
          2.At the end of the file, add
          
              export PATH={your nnictl path}:$PATH
              
            save and exit.
            
          3.source /etc/profile 
          
     
## To start using NNI CTL

please reference to the [NNI CTL document].


[NNI CTL document]: ../docs/NNICTLDOC.md