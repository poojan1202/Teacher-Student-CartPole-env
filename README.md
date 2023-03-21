
# Teacher-Student-CartPole-env

To implement Teacher-Student Paradigm ([Learning by Cheating](https://arxiv.org/abs/1912.12294) Framework) on a simpler gym environment.

Environment = Gym's `CartPole-v1`
Max timesteps = `500`


## Training Teacher Network
- To train the teacher network, cleanRL's PPO algorithm for discrete actions has been used with hyperparameters set as:
- | Hyper Parameter | Value |
    | -------- | -------- |
    | Learning Rate     | 3e-4
    | Training steps     | 300000     |
    | num-envs     | 4     |
    | gamma     | 0.99     |
    | lambda     | 0.95     |

- Both, The actor and critic network has 2 hidden layers with 64 nodes each. While the critic estimates the Value of the state, the actor predicts the most probable action.

#### Training Results
<p align="center">
<img src="https://i.imgur.com/mslgKAG.jpg" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/GzSEDGN.jpg" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/EleDs6b.jpg" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/FexAdHo.jpg" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/6kHIFJW.jpg" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/5DYkCJB.jpg" width="600" height="400" align="Center">
</p>


## Training Student Network

### Fully Observable Student
- For the case where the student's observation is same as the teacher's, that is, fully observable.
- Trained a Neural Network 2 hidden layers with 64 and 32 nodes respt. and output `layer with nodes = env.action_space`
- The student network is updated in supervised fashion using the actions generated by the trained teacher network
- Hyperparameters set while training:
- | Hyper Parameter | Value |
    | -------- | -------- |
    | Learning Rate     | 1e-4
    | Training episodes     | 2500     |
    | Memory Buffer     | 50000     |
    | Batch Size     | 128     |

- Note that the Memory Buffer is emptied after every 500 episodes so that the student network trains on more recent transitions made in the environment.

#### Training Results

<p align="center">
<img src="https://i.imgur.com/XElwyrt.png" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/SRxGTHh.png" width="600" height="400" align="Center">
</p>
<p align="center">
<img src="https://i.imgur.com/wrCJOkW.png" width="600" height="400" align="Center">
</p>


