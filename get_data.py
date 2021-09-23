from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('rtatman/chocolate-bar-ratings', path='data', unzip=True)
