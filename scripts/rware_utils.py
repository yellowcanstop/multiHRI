from stable_baselines3.common.vec_env import DummyVecEnv
from rware_wrapper import RwareToOAIWrapper

def get_rware_env(args):
    env_id = args.layout_names[0] 

    env_kwargs = {
        "args": args, # This will be popped by our wrapper
    }
    
    # 1. Create a function that returns your wrapped env
    def make_env():
        return RwareToOAIWrapper(env_id, **env_kwargs)

    # 2. Wrap it in a DummyVecEnv (it expects a list of functions)
    venv = DummyVecEnv([make_env])
    
    # 3. Add the layout name attribute so env_method can find it
    # We set it on the unwrapped inner environment
    venv.set_attr("layout_name", env_id)
    
    return venv