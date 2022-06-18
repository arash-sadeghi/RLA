# RLA repository
- This repository contains simulation code for the paper **Reinforcement Learning-based Aggregation for Robot Swarms**.
- Scripts inside folder *MAIN SIMULATOR FILES* are the main files for running simulations. 
- *DDPG* foldercontains the continues version of the paper. Its complete version is in another repository.

LBA-RL GitHub code has bug when landmark is on all 4 sides There are two sources for this bug both on act and reward:
  1) for landmarks less than 3 angle is added by 180
  2) calculated angle action can get negative sometimes which will cause the robot to stick to wall. In getQRs, false usage of allRobotIndx=np.delete(allRobotIndx,allRobotIndx==i) causes robot to sometimes not detect QRs
