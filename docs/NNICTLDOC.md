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
nnictl webui
nnictl experiment
nnictl config
nnictl log
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
	  | --webuiport, -w|  False| 8080|assign a port for webui|
  
      

* __nnictl resume__

  * Description
          
		  You can use this command to resume a stopped experiment.
       
  * Usage
	    
		nnictl resume [OPTIONS] 		
      Options:
     
      | Name, shorthand | Required|Default | Description |
      | ------ | ------ | ------ |------ |
    | --experiment, -e|  False| |ID of the experiment you want to resume|
  
     
      

* __nnictl stop__
  * Description
          
		  You can use this command to stop a running experiment.
       
  * Usage
	    	
        nnictl stop 
     
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
			
 	* __nnictl update concurrency__  
        * Description
          
		      You can use this command to update an experiment's concurrency.     
	  
	     * Usage
        
		       nnictl update concurrency [OPTIONS] 

            Options:
    
            | Name, shorthand | Required|Default | Description |
            | ------ | ------ | ------ |------ |
           | --value, -v|  True| |the number of allowed concurrent trials|
      	
     * __nnictl update duration__  
        * Description
        
               You can use this command to update an experiment's concurrency.  
		
		* Usage
			    	    
			    nnictl update duration [OPTIONS] 

          Options:
    
          | Name, shorthand | Required|Default | Description |
          | ------ | ------ | ------ |------ |
          | --value, -v|  True| |the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days.|
     

* __nnictl trial__
  * __nnictl trial ls__
    * Description
          
		    You can use this command to show trial's information.
   
     * Usage
  
           nnictl trial ls

  * __nnictl trial kill__
      * Description
            
			You can use this command to kill a trial job.
	   * Usage
  
              nnictl trial kill [OPTIONS] 
    
	      Options:  
	        
          | Name, shorthand | Required|Default | Description |
          | ------ | ------ | ------ |------ |
         | --trialid, -t|  True| |ID of the trial you want to kill.|      
     
      
          

### Manage WebUI
* __nnictl webui start__
     * Description
     
           Start web ui function for nni, and will get a url list, you can open any of the url to see nni web page.
      
     * Usage    
		  
		    nnictl webui start [OPTIONS]        

         Options:
    
         | Name, shorthand | Required|Default | Description |
         | ------ | ------ | ------ |------ |
       | --port, -p|  False| 8080|assign a port for webui|
     


* __nnictl webui stop__  
    * Description
             
			 Stop web ui function, and release url occupied. If you want to start again, use 'nnictl start webui' command
     * Usage
		    
			nnictl webui stop 
			
* __nnictl webui url__  
    * Description
             
			 Show the urls of web ui.
     * Usage
		    
			nnictl webui url

        
         


### Manage experiment information

* __nnictl experiment show__
  * Description
      
	     Show the information of experiment.
   * Usage
     
	     nnictl experiment show

 

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
     