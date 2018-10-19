nnictl
===
## Introduction
__nnictl__ is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc.

## Commands
nnictl support commands:
```
nnictl create
nnictl stop
nnictl update
nnictl resume
nnictl trial
nnictl experiment
nnictl config
nnictl log
nnictl webui
```
### Manage an experiment
* __nnictl create__ 
   * Description 
	    
		  You can use this command to create a new experiment, using the configuration specified in config file. 
          After this command is successfully done, the context will be set as this experiment, 
          which means the following command you issued is associated with this experiment, 
          unless you explicitly changes the context(not supported yet). 
   
  * Usage
  
        nnictl create [OPTIONS] 
	  
       Options:  
    
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
      | --config, -c|  True| |yaml configure file of the experiment|
      | --port, -p  |  False| |the port of restful server| 

* __nnictl resume__

  * Description
          
		  You can use this command to resume a stopped experiment.
       
  * Usage
	    
		nnictl resume [OPTIONS] 		
      Options:
     
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
    | --id, -i|  False| |ID of the experiment you want to resume|
  
     
      

* __nnictl stop__
  * Description
          
		You can use this command to stop a running experiment or multiple experiments.
       
  * Usage
	    	
        nnictl stop [id]
  
  * Detail
        
        1.If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message.
        2.If there is no id specified, and there is an experiment running, stop the running experiment, or print error message.
        3.If the id ends with *, nnictl will stop all experiments whose ids matchs the regular.
        4.If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.
        5.If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.
        6.Users could use 'nnictl stop all' to stop all experiments  
     
* __nnictl update__
    
	 * __nnictl update searchspace__
       * Description
          
		     You can use this command to update an experiment's search space.
       
       * Usage
 
              nnictl update searchspace [OPTIONS] 
         
            Options:
        
           | Name, shorthand | Required|Default | Description |
           | ------ | ------ | ------ |------ |
         | --filename, -f|  True| |the file storing your new search space|
         | --id, -i|  False| |ID of the experiment you want to set|
			
 	* __nnictl update concurrency__  
        * Description
          
		      You can use this command to update an experiment's concurrency.     
	  
	     * Usage
        
		       nnictl update concurrency [OPTIONS] 

            Options:
    
            | Name, shorthand | Required|Default | Description |
            | ------ | ------ | ------ |------ |
           | --value, -v|  True| |the number of allowed concurrent trials|
           | --id, -i|  False| |ID of the experiment you want to set|
      	
     * __nnictl update duration__  
        * Description
        
               You can use this command to update an experiment's concurrency.  
		
		* Usage
			    	    
			    nnictl update duration [OPTIONS] 

          Options:
    
          | Name, shorthand | Required|Default | Description |
          | ------ | ------ | ------ |------ |
          | --value, -v|  True| |the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days.|
          | --id, -i|  False| |ID of the experiment you want to set|
     

* __nnictl trial__
  * __nnictl trial ls__
    * Description
          
		    You can use this command to show trial's information.
   
     * Usage
  
           nnictl trial ls

      Options:
     
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
    | --id, -i|  False| |ID of the experiment you want to set|

  * __nnictl trial kill__
      * Description
            
			You can use this command to kill a trial job.
	   * Usage
  
              nnictl trial kill [OPTIONS] 
    
	      Options:  
	        
          | Name, shorthand | Required|Default | Description |
          | ------ | ------ | ------ |------ |
         | --trialid, -t|  True| |ID of the trial you want to kill.| 
         | --id, -i|  False| |ID of the experiment you want to set|     
     
      
          

### Manage experiment information

* __nnictl experiment show__
  * Description
      
	     Show the information of experiment.
   * Usage
     
	     nnictl experiment show
    
      Options:
      
        | Name, shorthand | Required|Default | Description |
        | ------ | ------ | ------ |------ |
      | --id, -i|  False| |ID of the experiment you want to set|


* __nnictl experiment status__
  * Description
      
	     Show the status of experiment.
   * Usage
     
	     nnictl experiment status
      
      Options:
     
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
     | --id, -i|  False| |ID of the experiment you want to set|


* __nnictl experiment list__
  * Description
      
	     Show the id and start time of all running experiments.
   * Usage
     
	     nnictl experiment list

 

* __nnictl config show__
    * Description
             
		    Display the current context information.
   
    * Usage
    
	      nnictl config show
        
  
### Manage log
* __nnictl log stdout__
   * Description
     
	     Show the stdout log content. 
   
   * Usage
         
		    nnictl log stdout [options]
        
    	Options:
    	
       | Name, shorthand | Required|Default | Description |
       | ------ | ------ | ------ |------ |
     | --head, -h| False| |show head lines of stdout|
     | --tail, -t|  False| |show tail lines of stdout|
	   | --path, -p|  False| |show the path of stdout file|
     | --id, -i|  False| |ID of the experiment you want to set|
	 
* __nnictl log stderr__
  * Description
  
        Show the stderr log content. 
  
  * Usage
  
        nnictl log stderr [options]
        
	   Options:
	   
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
    | --head, -h| False| |show head lines of stderr|
    | --tail, -t|  False| |show tail lines of stderr|
	  | --path, -p|  False| |show the path of stderr file|
    | --id, -i|  False| |ID of the experiment you want to set|

* __nnictl log trial__
  * Description
  
        Show trial log path. 
  
  * Usage
  
        nnictl log trial [options]
        
	   Options:
	   
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
    | --id, -I| False| |the id of trial|


### Manage webui
* __nnictl webui url__
   * Description
     
	     Show the urls of the experiment. 
   
   * Usage
         
		    nnictl webui url
        
    	Options:
    	
       | Name, shorthand | Required|Default | Description |
       | ------ | ------ | ------ |------ |
     | --id, -i|  False| |ID of the experiment you want to set|