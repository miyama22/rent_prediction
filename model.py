import polars as pl
import pandas as pd
import lightgbm as lgb
import gc
import pickle
import xgboost as xgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
from tqdm import tqdm

import numpy as np

def lightgbm_training(x_train: pl.DataFrame, y_train:pl.DataFrame, x_valid:pl.DataFrame, y_valid:pl.DataFrame
                    ,verbose , categorical_feature, early_stopping_round, regression_lgb_params, num_boost_round):
    lgb_train = lgb.Dataset(x_train.to_pandas(), y_train.to_pandas())
    lgb_valid = lgb.Dataset(x_valid.to_pandas(), y_valid.to_pandas())
    
    model = lgb.train(
        params=regression_lgb_params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        categorical_feature=categorical_feature,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_round, verbose=verbose),
                lgb.log_evaluation(verbose),
                ]
    )
    
    valid_pred = model.predict(x_valid)
    return model, valid_pred

def xgboost_training(x_train: pl.DataFrame, y_train: pl.DataFrame, x_valid: pl.DataFrame, y_valid: pl.DataFrame,
                    regression_xgb_params, num_boost_round, early_stopping_round, verbose):
    
    feature_names = x_train.columns
    
    xgb_train = xgb.DMatrix(data=x_train.to_pandas(), label=y_train.to_pandas(), feature_names=feature_names)
    xgb_valid = xgb.DMatrix(data=x_valid.to_pandas(), label=y_valid.to_pandas(), feature_names=feature_names)
    model = xgb.train(
                regression_xgb_params,
                dtrain = xgb_train,
                num_boost_round = num_boost_round,
                evals = [(xgb_train, 'train'), (xgb_valid, 'eval')],
                early_stopping_rounds = early_stopping_round,
                verbose_eval = verbose
            )
    # Predict validation
    valid_dmatrix = xgb.DMatrix(
        data=x_valid.to_pandas(),
        feature_names=feature_names
    )
    valid_pred = model.predict(valid_dmatrix)
    return model, valid_pred

def catboost_training(x_train: pl.DataFrame, y_train: pl.DataFrame, x_valid: pl.DataFrame, y_valid: pl.DataFrame,
                    regression_cat_params, early_stopping_round, verbose):
    cat_train = Pool(data=x_train.to_pandas(), label=y_train.to_pandas())
    cat_valid = Pool(data=x_valid.to_pandas(), label=y_valid.to_pandas())
    model = CatBoostRegressor(**regression_cat_params)
    model.fit(cat_train,
            eval_set = [cat_valid],
            early_stopping_rounds = early_stopping_round,
            verbose = verbose,
            use_best_model = True)
    # Predict validation
    valid_pred = model.predict(x_valid.to_pandas())
    return model, valid_pred


def gradient_boosting_model_cv_training(method: str, train_df: pl.DataFrame, features: list, n_folds, target_col,
                                        MODEL_DATA_PATH, seed, VER, OOF_DATA_PATH, verbose,
                                        categorical_feature, early_stopping_round, regression_lgb_params, num_boost_round,
                                        regression_xgb_params, regression_cat_params):
    oof_predictions = np.zeros(train_df.height)
    oof_fold = np.zeros(train_df.height)
    
    for fold in tqdm(range(n_folds)):
        print(f'start fold{fold}')
        
        #foldごとにtrainとvalidに分ける
        train_fold = train_df.filter(pl.col('fold') != fold)
        valid_fold = train_df.filter(pl.col('fold') == fold)
        
        #説明変数と目的変数に分割
        x_train = train_fold.select(features)
        x_valid = valid_fold.select(features)
        y_train = train_fold.select(target_col)
        y_valid = valid_fold.select(target_col)
        
        #学習
        if method == 'lightgbm':
            model, valid_pred = lightgbm_training(x_train, y_train, x_valid, y_valid, verbose, categorical_feature = categorical_feature, 
                                                early_stopping_round=early_stopping_round, regression_lgb_params=regression_lgb_params, num_boost_round=num_boost_round)
        if method == 'xgboost':
            model, valid_pred = xgboost_training(x_train, y_train, x_valid, y_valid, regression_xgb_params=regression_xgb_params, 
                                                num_boost_round=num_boost_round, early_stopping_round=early_stopping_round, verbose=verbose)

        if method == 'catboost':
            model, valid_pred = catboost_training(x_train, y_train, x_valid, y_valid, regression_cat_params=regression_cat_params,
                                                early_stopping_round= early_stopping_round, verbose=verbose)
            
        #モデルの保存
        pickle.dump(model, open(MODEL_DATA_PATH / f'{method}_fold{fold + 1}_seed{seed}_ver{VER}.pkl', 'wb'))
        
        #oofに追加
        is_valid = train_df.get_column('fold') == fold
        oof_predictions[is_valid] = valid_pred
        oof_fold[fold] = fold
        
    score = np.sqrt(mean_squared_error(train_df[target_col], oof_predictions))
    print(f'{method} our out of folds CV rmse is {score}')
    
    # OOFを作成
    oof_df = pl.DataFrame({target_col:train_df.get_column(target_col), f'{method}_predictions':oof_predictions, 'fold':oof_fold})
    oof_df.write_csv(OOF_DATA_PATH / f'oof_{method}_seed{seed}_ver{VER}.csv')


def Learning(input_df: pl.DataFrame, features: list, METHOD_LIST,  n_folds, target_col,
                                        MODEL_DATA_PATH, seed, VER, OOF_DATA_PATH, verbose,
                                        categorical_feature, early_stopping_round, regression_lgb_params, num_boost_round,
                                        regression_xgb_params, regression_cat_params):
    for method in tqdm(METHOD_LIST):
        gradient_boosting_model_cv_training(method, input_df, features, n_folds=n_folds, target_col=target_col, MODEL_DATA_PATH=MODEL_DATA_PATH,
                                            seed=seed, VER=VER, OOF_DATA_PATH=OOF_DATA_PATH, verbose=verbose, categorical_feature=categorical_feature,
                                            early_stopping_round=early_stopping_round, regression_lgb_params=regression_lgb_params, num_boost_round=num_boost_round,
                                            regression_xgb_params=regression_xgb_params, regression_cat_params=regression_cat_params)


def lightgbm_inference(x_test: pl.DataFrame, n_folds, MODEL_DATA_PATH, seed, VER):
    test_pred = np.zeros(x_test.height)
    x_test = x_test.to_pandas()
    for fold in range(n_folds):
        model = pickle.load(open(MODEL_DATA_PATH / f'lightgbm_fold{fold + 1}_seed{seed}_ver{VER}.pkl', 'rb'))
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / n_folds

def xgboost_inference(x_test: pl.DataFrame, n_folds, MODEL_DATA_PATH, seed, VER):
    test_pred = np.zeros(x_test.height)
    x_test = x_test.to_pandas()
    for fold in range(n_folds):
        model = pickle.load(open(MODEL_DATA_PATH / f'xgboost_fold{fold + 1}_seed{seed}_ver{VER}.pkl', 'rb'))
        # Predict
        pred = model.predict(xgb.DMatrix(x_test))
        test_pred += pred
    return test_pred / n_folds

def catboost_inference(x_test: pl.DataFrame, n_folds, MODEL_DATA_PATH, seed, VER):
    test_pred = np.zeros(x_test.height)
    x_test = x_test.to_pandas()
    for fold in range(n_folds):
        model = pickle.load(open(MODEL_DATA_PATH / f'catboost_fold{fold + 1}_seed{seed}_ver{VER}.pkl', 'rb'))
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / n_folds

def gradient_boosting_model_inference(method: str, test_df: pl.DataFrame, features: list, n_folds, MODEL_DATA_PATH, seed, VER):
    x_test = test_df.select(features)
    if method == 'lightgbm':
        test_pred = lightgbm_inference(x_test, n_folds=n_folds, MODEL_DATA_PATH=MODEL_DATA_PATH,
                                    seed=seed, VER=VER)
    if method == 'xgboost':
        test_pred = xgboost_inference(x_test, n_folds=n_folds, MODEL_DATA_PATH=MODEL_DATA_PATH,
                                    seed=seed, VER=VER)
    if method == 'catboost':
        test_pred = catboost_inference(x_test, n_folds=n_folds, MODEL_DATA_PATH=MODEL_DATA_PATH,
                                    seed=seed, VER=VER)
    return test_pred

def Predicting(input_df: pl.DataFrame, features: list, METHOD_LIST, model_weight_dict, n_folds, MODEL_DATA_PATH, seed, VER):
    output_df = input_df.clone()
    output_df = output_df.with_columns(pl.lit(0).alias('pred')).to_pandas()
    # CFG.METHOD_LISTの各methodに対して推論を行い、その結果を出力データフレームに追加
    for method in tqdm(METHOD_LIST):
        output_df[f'{method}_pred'] = gradient_boosting_model_inference(method, input_df, features, n_folds=n_folds, MODEL_DATA_PATH=MODEL_DATA_PATH,
                                    seed=seed, VER=VER)
        output_df['pred'] += model_weight_dict[method] * output_df[f'{method}_pred']
    return output_df

# 後処理を行う関数
def after_treatment(test_input:pl.DataFrame, model_pred_weight:float, train_id_weight):
    assert model_pred_weight + train_id_weight == 1, '重みの合計が1ではありません'

    out_put_series = (
        test_input
        .with_columns((
            # モデルの予測値から、部屋の賃料を求める
            pl.col('pred') * pl.col('room_area')).cast(pl.Int64).alias('pred_room_money'),
            # ビルごとの平均1平米賃料から、部屋の賃料を求める
            (pl.col('money_per_1m2_mean_building') * pl.col('room_area')).cast(pl.Int64).alias('room_money_from_building_mean1m2')
            )
        # idがtrainにあるものは重みづけを行い、ないものは予測値をそのまま用いる
        .with_columns(
            pl.when(pl.col('money_per_1m2_mean_building').is_not_nan())
            .then(((pl.col('pred_room_money') * model_pred_weight) + (pl.col('room_money_from_building_mean1m2') * train_id_weight)).alias('submit_pred'))
            .otherwise(pl.col('pred_room_money').alias('submit_pred'))
        )
    )
    return out_put_series