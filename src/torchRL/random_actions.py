import gymnasium as gym
import time

env = gym.make("intersection-multi-agent-v0", render_mode="rgb_array")

env.configure({
    "vehicles_count": 5,
    "controlled_vehicles": 3

})

env.reset()
done = False
env.render()

# sleep for 5 seconds
time.sleep(2)

# do 10 steps
for i in range(10):
    test = env.step(env.action_space.sample())
    env.render()
    time.sleep(1)

env.close()
