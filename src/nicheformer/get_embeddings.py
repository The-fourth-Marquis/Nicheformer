import pprint
import wandb
from ._embeddings import get_embeddings_organ
from .config_files import _config_embeddings as config

if __name__ == "__main__":
        
    get_embeddings_organ(config=config.sweep_config)
    


    
    


    