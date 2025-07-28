## Setting up a virtual env 
Use of a virtual environment for running the python scripts is recommended.   
To do this (using conda), in the command-line (or terminal) in your local directory for this repository run: 
```
conda create -n pigmm python=3.9
```

Then activate this new environment and install the required packages:  
```
conda activate pigmm && pip install -r environment/requirements.txt
```
