

import argparse
import utils

import datetime

SHOW_PLOT = False
SAVE_PLOT = True
RESHUFFLE = False
AUGMENT_TRAIN = False
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 40
PLOT_FREQ = 100
EVAL_FREQ = 100
BACKUP = False
BACKUP_FREQ = -1


def count_params(config):
    
    _, n_inputs, n_outputs = utils.build_dataloaders(
        config.dataset,
        config.batch_size,
        config.workers,
        config.augment,
        config.earlystop,
        config.reshuffle
    )
    expname = utils.exp_name(config)
    
    model = utils.build_model(config, n_inputs, n_outputs)
    
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    totmemory = sum([p.numel() * p.element_size() for p in model.parameters() if p.requires_grad])
    totmemory += sum([p.numel() * p.element_size() for p in model.buffers()])
    totmemory //= 1024 ** 2
    
    print("Total Parameters: {:<15} | Total Memory (MB): {:<15}".format(nparams, totmemory))
    
    # for i, (name, mod) in enumerate(model.named_modules()):
        # print("\t", i, el.__class__, el.in_type.size, el.out_type.size)

        # mem = sum([p.numel() * p.element_size() for p in mod.parameters(recurse=False)])
        # mem += sum([p.numel() * p.element_size() for p in mod.buffers(recurse=False)])

        # mem //= 1024**2
        
        # print(f"\t{i}: {name}", mod, mem)
        
    return expname, nparams


################################################################################
################################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    ######## EXPERIMENT'S PARAMETERS ########
    parser = utils.args_exp_parameters(parser)

    config = parser.parse_args()
    
    print("----------------------------------------------------------")
    print(datetime.datetime.now())
    
    expname, nparams = count_params(config)

    print(f"{expname}:\t{nparams} parameters")
    
    print(datetime.datetime.now())
    print("----------------------------------------------------------")
