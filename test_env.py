state = env.reset()
print("Initial State:", state)

for _ in range(5):
    action = env.action_space.sample() 
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        break
