# RLA repository
- This repository contains simulation code for the paper **Reinforcement Learning-based Aggregation for Robot Swarms**.
- Scripts inside folder *MAIN SIMULATOR FILES* are the main files for running simulations. 
- Entry point is ```MAIN_SIMULATOR_FILES/MAIN.py```. For using this simulator only change params.json file **DO NOT CHANGE THE SOURCE CODE**
- You can install dependencies from requriemetns.txt
# lessons learned:
- dont write all functionalities in one code. seperate them to modules or branches even. This code encapsulate all functionalities in one script and its hard to deal with it.
- If you create an venv in a folder then change the name of that folder, running ``` venv/bin/activate ``` will look like its working but it will point to global interpretet. If you check the ```activate``` you will see the path is hard coded there. [stackoverflow link](https://stackoverflow.com/questions/65390129/venv-activate-doesnt-not-change-my-python-path)
- seperate experiment cases with unique codes.
- for dependencies and virtual environments, use pipreqs. If you have ROS in your PC, pip freeze will add ROS packages as well. However, pipreqs will only say the used packages.


