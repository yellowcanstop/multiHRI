from rware_wrapper import RwareToOAIWrapper

def get_rware_env(args):
    env_id = args.layout_names[0] 
    return RwareToOAIWrapper(env_id)