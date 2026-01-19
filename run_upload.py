from upload.upload import upload_model, upload_dataset
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument('--is_private', action='store_true')

    args = parser.parse_args()
    lang_name = args.language
    username = args.username
    is_private = args.is_private

    upload_model(lang_name, username, is_private)
    upload_dataset(lang_name, username, is_private)