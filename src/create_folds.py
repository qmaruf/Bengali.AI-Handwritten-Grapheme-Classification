import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    print ('hello world')
    df = pd.read_csv('../input/train.csv')
    print (df.head())
    df.loc[:, 'kfold']=-1
    
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.image_id.values
    print (df.columns)
    # exit()
    y = df[['grapheme_root', "vowel_diacritic", "consonant_diacritic"]].values
    
    mskf = MultilabelStratifiedKFold(n_splits=5)
    
    for fold, (train_, val_) in enumerate(mskf.split(X, y)):
        print ("Train:", train_, "val", val_)
        df.loc[val_, "kfold"] = fold
    
    print (df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv", index=False)