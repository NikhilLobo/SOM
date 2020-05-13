# SOM
Project Title:			
   Python implementation of Kohonen Self Organizing Maps (KSOM) and Constraint Topological Maps (CTM) .
   Prerequisites
IDE:
   Spyder

Programming Languages:
	 Python

External Libraries:
	 Graphics.py
Versioning
   Git is used for versioning.

How to run the code.
  -> Clone the repository from the github.
  -> Goto the directory which has been downloaded.
  -> In this project it is SOM/ directory and run the below command:
      python SOM.py
  -> Small GUI window will open after executing the SOM.py.
  -> Feed various hyper parameters in the GUI window.
      o Dataset file name :
        The SOM/ directory also has a directory called data/ for the dataset. It has data fille(data.csv) generated randomly.         Just enter the data as input don't add the extension .csv in the window
      o Dimension of the map:
        The dimension of the map needs to be either 1 or 2. That is used for the type of map needed to plot. 
      o Number of nodes : 
        Once selected 1 or 2 dimensional map. Next step is setting the number of nodes in that map. For 2                             dimensional map it will consider square for example if entered value is 5 internally it will consider it as 5*5 map.           So don't give larger values when selecting a 2D map.
      o Learning Rate: 
        To shrink the neighborhood eventually. Enter the learning rate like 0.5.
      o Epochs: 
        Enter the number of iterations.
      o Animation: 
        Select either Yes or No for the animation of the map.
      o Train Map Button: 
        Click it before visualizing so that all the parameters are set and trained before visualization.
      o Dependent value: 
        Select either None or one of the dependent feature names.
      o Feature 1: 
        Select the feature to plot on X-axis
      o Feature 2: 
        Select the feature to plot on Y-axis
      o Visualize: Click visualize to see the map on the new window.

Author:
	Nikhil Lobo, Graduate student, University of Wisconsin Riverfalls.

 
