import os
from importlib import import_module
from pathlib import Path

from src.mmodel.basic_params import basic_parser


# 将basic_parser所有的参数全部设置好
def get_basic_params():
    config_path = Path(f"./mmodel/ZZZ_model/{os.environ['config_file']}")
    if not config_path.exists():
        raise Exception("{} should be config file".format(config_path))

    save_path = Path("./mmodel/{}/__saved_model__".format(model))
    save_path.mkdir(exist_ok=True)
    os.environ["SVAE_PATH"] = str(save_path)

    log_path = Path("./mmodel/{}/__log__".format(model))
    log_path.mkdir(exist_ok=True)
    os.environ["SVAE_LOG"] = str(log_path)

    basic_parser._default_config_files.append(config_path)
    params, _ = basic_parser.parse_known_args()
    return params


def get_model(**kwargs):
    dir_path = Path("./mmodel/ZZZ_model" )
    model = import_module("src.mmodel.ZZZ_model").model(**kwargs)
    return model
