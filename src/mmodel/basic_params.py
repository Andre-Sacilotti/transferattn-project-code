import configargparse

basic_parser = configargparse.ArgParser(
    default_config_files=["mmodel/basic_config.yml"]
)


"""
gpu config
"""

basic_parser.add("--gpu", type=str)

basic_parser.add("--num_workers", type=int)

basic_parser.add("--random_seed", type=int)

"""
training setting
"""

basic_parser.add("--eval_epoch_interval", type=int)

basic_parser.add("--epoch", type=int)

"""
params for logging metrics comet.ml
"""

basic_parser.add("--enable_log", action="store_true")

basic_parser.add("--log_step_interval", type=int)

basic_parser.add("--comet_api_key", type=str)

basic_parser.add("--wandb_api_key", type=str)

basic_parser.add("--project_name", type=str)
