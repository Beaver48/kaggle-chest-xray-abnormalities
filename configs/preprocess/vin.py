import configs.preprocess.prep as prep

config = dict(
    train_metapath='data/raw/vinbigdata/train.csv',
    train_images_dir='data/raw/vinbigdata/train/',
    test_images_dir='data/raw/vinbigdata/test/',
    clear=True,
    visualize=True,
    aggree_rad=False,
    result_dir='data/processed/vinbigdataVOC2012',
    preprocessor=prep.preprocessor)
