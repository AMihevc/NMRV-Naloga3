# NMRV-Naloga3
Work on 3rd assigment for NMRV 

For this to work you need a toolkit from the instructions. I used the toolkit-lite which is a lite version of the VOT toolkit. 
To run the code setup the workdir and toolkit as described in the instructions. 
Then run the tracker with the following command: 
```
python .\evaluate_tracker.py --workspace_path C:\Users\am8130\Documents\Faks\2.semester\NMRV\3.naloga\NMRV-Naloga3\workspace-dir --tracker mosse_simple_tracker
```
The tracker will run on the dataset and the results will be saved in the results folder.


To calculate more consice results run the following command: 
```
python .\calculate_measures.py --workspace_path C:\Users\am8130\Documents\Faks\2.semester\NMRV\3.naloga\NMRV-Naloga3\workspace-dir --tracker mosse_simple_tracker
```
