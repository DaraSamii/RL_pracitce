import gym


env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, directory = outdir, force = True)
env.reset()
env.render(mode = 'human')
'''
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    state, reward, done = env.step(action)
    #print(a)
env.close()'''