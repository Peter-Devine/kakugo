from train.make_train_data import make_gen_trans_full_train_data
from train.make_train_configs import make_train_config
from utils.utils import filesafe_parse_name, read_yaml
import argparse

def main(language, config_data):
    language = filesafe_parse_name(language)

    make_gen_trans_full_train_data(language, config_data)

    make_train_config(language, config_data)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="./config.yaml")

    args = parser.parse_args()
    language = args.language

    config_data = read_yaml(args.config_path)

    main(language, config_data)