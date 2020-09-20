import gym

from phillRL import Agent


if __name__ == '__main__':
    env = gym.make('LunerLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size = 64, n_actions = 4,
                  eps_end = 0.01, input_dims = [8], lr = 0.003)

    scores, eps_history = [],[]
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
