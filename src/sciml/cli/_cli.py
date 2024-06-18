import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

class SCIMLCli(LightningCLI):
    
    def __init__(self, **kwargs):
        
        if not 'parser_kwargs' in kwargs:
            kwargs['parser_kwargs'] = {}
            
        kwargs['parser_kwargs'].update({
            "default_env": True, 
            "parser_mode": "omegaconf"
        })
            
        from sciml import VAEModel
        super().__init__(
            model_class=VAEModel, 
            subclass_mode_data=True, 
            subclass_mode_model=True,
            save_config_callback=None,
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.link_arguments('data.init_args.batch_size', 'model.init_args.batch_size')
        
    def before_fit(self):
        self.trainer.logger.log_hyperparams(self.config)