import gym
import torch
import torch.nn as nn
import numpy as np
import statistics
from PIL import Image

env = gym.make("CartPole-v0")
NUM_EPISODES = 1000
total_rewards = 0.0
rewards = []

models = ["agent1.dat", "agent2.dat", "agent3.dat", "agent4.dat", "agent5.dat"]
nets = []

#frames = []

# def frames_to_img(frames):
#     for i in range(len(frames)):
#         img = Image.fromarray(frames[i])
#         img_name = 'ensembling_' +  str(i) + '.png'
#         img.save(img_name)

class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

for i in range(len(models)):
    nets.append(DQN(env.observation_space.shape[0], env.action_space.n))
    nets[i].load_state_dict('models/' + torch.load(models[i]))

if __name__ == "__main__":
    for i in range(NUM_EPISODES):
        state = env.reset()
        # env.render()
        # frames.append(env.render(mode="rgb_array"))
        episode_rewards = 0
        is_done = False
        while not is_done:
            state_a = np.array([state], copy=False)
            state_v = torch.FloatTensor(state_a)
            actions = []
            for j in range(len(nets)):
                q_vals_v = nets[j](state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())
                actions.append(action)
            # actions_string = 'actions: ' + str(actions)
            # print(actions_string + ', action: %d' % (statistics.mode(actions)))
            state, reward, is_done, _ = env.step(statistics.mode(actions))
            # env.render()
            # frames.append(env.render(mode="rgb_array"))
            episode_rewards += reward
        if i % 10 == 0:
            print("episode: %d, reward: %d" % (i, episode_rewards))
        rewards.append(episode_rewards)
        total_rewards += episode_rewards
    env.close()
    print("mean reward: %d" % (total_rewards/NUM_EPISODES))
    # frames_to_img(frames)
    # np.savetxt("ensembling_rewards.txt", rewards, fmt="%d")