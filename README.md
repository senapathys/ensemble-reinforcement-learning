# Improving Reinforcement Learning Model Performance with a Multi-Agent Ensemble Algorithm

This is a Python application of the majority voting ensemble system described in the 2008 paper "Ensemble Algorithms in Reinforcement Learning" to the CartPole optimal control environment.

Five individual Deep Q-Network (DQN) RL models were trained until near-convergence on the CartPole-v0 environment provided by OpenAI gym. These trained models were tested (zero exploration) and performance was noted. These models were then used to compose and test an ensemble system.

The results are below. As evident, the ensemble algorithm outperforms every individual model.

![rewards](https://user-images.githubusercontent.com/47801356/88360752-7ef5ef00-cd3c-11ea-9639-5e200ec40d2b.png)



