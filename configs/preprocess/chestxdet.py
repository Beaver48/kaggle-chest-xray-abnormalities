import configs.preprocess.prep as prep

config = dict(
    metapath='data/raw/chestxdet10/*.json',
    images='data/raw/chestxdet10/*/*',
    clear=False,
    visualize=False,
    result_dir='data/processed/vin_dataVOC2012',
    preprocessor=prep.preprocessor)