import configs.preprocess.prep as prep

config = dict(
    bbox_metapath='data/raw/nihdata/BBox_List_2017.csv',
    images_regex='data/raw/nihdata/image*/*/*',
    clear=False,
    result_dir='data/processed/vin_dataVOC2012',
    preprocessor=prep.preprocessor)
