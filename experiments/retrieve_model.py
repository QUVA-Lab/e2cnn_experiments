import argparse
import utils
import torch

from collections import OrderedDict

###############################################################################################
# retrieve a stored model
# use the usual command line parameters to specify the experiment that generated such model
# add the "--seed <SEED>" to specify the seed used to run that experiment 
# in order to discriminate between different run of the same experiments
###############################################################################################

def retrieve(config):
    
    path = utils.backup_path(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # N.B.: the saved file contains the model wrapped in DataParallel
    
    if device == "cpu":
        state_dict = torch.load(path, map_location=device)
    else:
        state_dict = torch.load(path)

    _, n_inputs, n_outputs = utils.build_dataloaders(
        config.dataset,
        config.batch_size,
        config.workers,
        config.augment,
        config.earlystop,
        config.reshuffle
    )
    model = utils.build_model(config, n_inputs, n_outputs)
    
    # filter out `module.` prefix (coming from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    del state_dict

    model = model.to(device)
    # load params
    model.load_state_dict(new_state_dict)
    
    return model


if __name__ == "__main__":
    
    # Parse training configuration
    parser = argparse.ArgumentParser()
    
    parser = utils.args_exp_parameters(parser)

    parser.add_argument('--seed', type=int, help='Seed of the experiment')
    parser.add_argument('--output', type=str, default=None, help="Path where to store the extracted model")
    
    config = parser.parse_args()
    
    # Train the model
    model = retrieve(config)
    
    if config.output is not None:
        torch.save(model.state_dict(), config.output)
