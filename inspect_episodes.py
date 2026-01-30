
import habitat
from config_utils import hm3d_config
import os
from habitat.config.read_write import read_write

print(f"Checking episodes in {os.getcwd()}")
try:
    config = hm3d_config(episodes=10)
    with read_write(config):
        config.habitat.environment.iterator_options.shuffle = False
    
    with habitat.Env(config=config) as env:
        print(f"Total episodes available: {len(env.episodes)}")
        for i in range(5):
             ep = env.episodes[i]
             print(f"Index {i}: ID={ep.episode_id}, Scene={ep.scene_id}, Object={ep.object_category}")

except Exception as e:
    print(f"Error: {e}")
