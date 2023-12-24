import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt

df_2013 = pd.read_csv('C:/Users/JunYeong.DESKTOP-IOMT4HU/Desktop/vscode/XGBoost/xgboost_data/xgboost_file_2013.csv')


optimized_list_2013 = ['n2e', 'n2a', 'n2b', 'n7a', 'c9b', 'i2b', 'production_emp_rate', 'c22b_1.0', 'b6b',
                       'b7', 'Finan_emp_rate', 'b5', 'Estate_emp_rate', 'hh_elec_t', 'Trans_emp_growth', 'k3bc', 'Hotel_emp_growth',
                       'b3', 'nbi_per_capita', 'l9b', 'edu_ter_15_num', 'f2', 'e11_1.0', 'c7', 'a4b_Retail', 'a4b_Food', 'a4b_Garments',
                       'a4b_Hotel and Restaurants', 'a4b_Textiles', 'a4b_Others', 'L1_NAME_Chittagong', 'L1_NAME_Dhaka', 'L1_NAME_Sylhet',
                       'L1_NAME_Rajshahi', 'L1_NAME_Khulna', 'a6b_Large', 'a6b_SME', 'Elec_emp_rate', 'Public_emp_rate', 'Education and Technology_3_sum_pop',
                      'all_pop']
drop_list_2013 = ['L1_CODE', 'L2_CODE','L2_NAME', 'L3_CODE', 'L3_NAME','d2', 'l1', 'z1_log', 'x1', 'l6', 'Rank','a2','a3a','a4a','a6a',
             'idstd', 'lat_mask', 'lon_mask', 'a6a', 'a0' , 'l5a', 'l5b', 'b6', 'b2d','b6', 'l3a', 'l3b']


##2013년
#학습데이터 테스트 데이터 분리
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#데이터 정리하기
y_target = df_2013['z1']
X_features = df_2013.drop(drop_list_2013,axis=1, inplace=False)
X_features = X_features.drop('z1',axis=1, inplace=False)
X_features = X_features[optimized_list_2013]
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val= train_test_split(X_train, y_train,
                                         test_size=0.1, random_state=42)

#평가지표 정의
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def get_mae(model):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print('{0} MAE: {1}'.format(model.__class__.__name__, np.round(mae, 3)))
    return mae

def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    print('{0} RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

def get_r2score(model):
    pred = model.predict(X_test)
    r2score = r2_score(y_test, pred) # 함수명을 r2score로 수정
    print('{0} R2_score: {1}'.format(model.__class__.__name__, np.round(r2score, 3))) # 변수명도 r2score로 수정
    return r2score
    
# 여러 모델들을 list 형태로 인자로 받아서 개별 모델들의 RMSE와 R^2을 list로 반환.
def get_maes(models):
    maes =[]
    for model in models:
        mae = get_mae(model)
        maes.append(mae)
    return maes

def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

def get_r2scores(models):
    r2scores = []
    for model in models:
        r2score = get_r2score(model)
        r2scores.append(r2score)
    return r2scores

print(X_features)

#XGBoost 순정
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

xgb_reg = XGBRegressor(n_estimators=1000)
evals4 = [(X_tr, y_tr), (X_val, y_val)]
xgb_reg.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric='rmse', 
                eval_set=evals4, verbose=True)

# 모델 간의 rmse, r2비교
models = [xgb_reg]
get_maes(models)
get_rmses(models)
get_r2scores(models)

#교차검증 kfold 5
from sklearn.model_selection import cross_val_score

X_tr1, X_val1, y_tr1, y_val1= train_test_split(X_features, y_target,
                                         test_size=0.2, random_state=42)

fit_params={'early_stopping_rounds': 50,
            'verbose': False,
            'eval_set': [[X_val1, y_val1]]}

def get_avg_rmse_r2_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행. 모델별 CV RMSE값과 평균 RMSE, R^2 출력
        mae_list = -cross_val_score(model, X_tr1, y_tr1,
                                             scoring='neg_mean_absolute_error', cv=5, fit_params = fit_params)
        rmse_list = np.sqrt(-cross_val_score(model, X_tr1, y_tr1,
                                             scoring="neg_mean_squared_error", cv=5, fit_params = fit_params))
        r2_list = cross_val_score(model, X_tr1, y_tr1,
                                  scoring="r2", cv=5, fit_params = fit_params)
        
        mae_avg = np.mean(mae_list)
        rmse_avg = np.mean(rmse_list)
        r2_avg = np.mean(r2_list)
        print('\n{0} CV MAE 값 리스트: {1}'.format(model.__class__.__name__, np.round(mae_list, 3)))
        print('\n{0} CV 평균 MAE 값 리스트: {1}'.format(model.__class__.__name__, np.round(mae_avg, 3)))
        print('\n{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))
        print('{0} CV R^2 값 리스트: {1}'.format(model.__class__.__name__, np.round(r2_list, 3)))
        print('{0} CV 평균 R^2 값: {1}'.format(model.__class__.__name__, np.round(r2_avg, 3)))

# 앞 예제에서 학습한 모델들의 CV RMSE값과 R^2 값을 출력
models = [xgb_reg]
print(get_avg_rmse_r2_cv(models))

#Bayesian Optimization 구간 설정
from hyperopt import hp

xgb_search_space = {
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0, 50),
    'reg_lambda': hp.uniform('reg_lambda', 10, 100),
    #'min_split_loss': hp.uniform('min_split_loss', 0, 50),
    #'max_leaves' : hp.quniform('max_leaves', 200, 250, 1),
    'learning_rate' : hp.uniform('learning_rate', 0, 0.3),
    'min_child_weight' : hp.uniform('min_child_weight', 0, 50)
}

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from hyperopt import STATUS_OK
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

xgb_reg = XGBRegressor()
models=[xgb_reg]

#fit_params={'early_stopping_rounds': 50,
            #'verbose': False,
            #'eval_set': [[X_val, y_val]]}

def objective_func3(search_space):
    xgb_reg = XGBRegressor( max_depth=int(search_space['max_depth']),
                            min_child_weight=search_space['min_child_weight'],
                            learning_rate=search_space['learning_rate'],
                            tree_method = 'exact',
                            n_estimators = 1000,
                            colsample_bytree=search_space['colsample_bytree'],
                            subsample=search_space['subsample'],
                            reg_alpha=search_space['reg_alpha'],
                            reg_lambda=search_space['reg_lambda'],
                            #min_split_loss=search_space['min_split_loss'],
                            eval_metric='rmse'
                          )
    
    RMSE = -cross_val_score(xgb_reg, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
    return {'loss': np.mean(RMSE), 'status': STATUS_OK}

#xgboost 튜닝 진행

from hyperopt import fmin, tpe, Trials

#Best parameter로 학습 진행
trial_val = Trials()
best_xgb = fmin(fn=objective_func3,
            space=xgb_search_space,
            algo=tpe.suggest,
            max_evals=100, 
            trials=trial_val)

print('best_xgb:', best_xgb)


best_xgb_2013 = XGBRegressor(n_estimators=1000, learning_rate=0.01, 
                        max_depth=8, min_child_weight=30.27,
                        colsample_bytree=0.79, subsample=0.95,
                        reg_alpha=5.54, reg_lambda=82.47#, min_split_loss=round(best_xgb['min_split_loss'], 2)
                        )
evals3 = [(X_tr, y_tr), (X_val, y_val)]
best_xgb_2013.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric='rmse', 
                eval_set=evals3, verbose=True)
                                                                     
models = [best_xgb_2013]
get_maes(models)
get_rmses(models)
get_r2scores(models)

results = best_xgb_2013.evals_result()

# plot learning curves
figure = pyplot.figure(figsize = (10,6))
pyplot.plot(results['validation_0']['rmse'], label='train')
pyplot.plot(results['validation_1']['rmse'], label='test')

# show the legend
pyplot.legend()
pyplot.xlabel('iterations')
pyplot.title('XGBoost loss function_2013')
pyplot.ylabel('RMSE')
# show the plot
pyplot.show()

#shapely value
import shap

shap.initjs()
explainer = shap.TreeExplainer(best_xgb_2013) #제일 잘 나온 모델 이용
shap_values = explainer.shap_values(X_features)

exclude_variables = ['a4b_Retail', 'a4b_Food','a4b_Garments','a4b_Hotel and Restaurants','a4b_Textiles', 'a4b_Others',
                    'L1_NAME_Chittagong', 'L1_NAME_Dhaka', 'L1_NAME_Sylhet','L1_NAME_Rajshahi', 'L1_NAME_Khulna',
                    'a6b_Large', 'a6b_SME']

included_columns = [col for col in X_features.columns if col not in exclude_variables]

# 그래프에서 제외할 변수의 인덱스를 찾기
exclude_indices = [X_features.columns.get_loc(col) for col in exclude_variables if col in X_features.columns]

# 필터링된 그래프를 생성하여 출력
filtered_shap_values = np.delete(shap_values, exclude_indices, axis=1)
shap.summary_plot(filtered_shap_values, X_features.drop(columns=exclude_variables), plot_type='bar', max_display=50)
shap.summary_plot(filtered_shap_values, X_features.drop(columns=exclude_variables), max_display=50)

feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
#feat_imp.to_csv('2013_최종SHAP.csv')

# force plot
shap.force_plot(explainer.expected_value, filtered_shap_values[1], X_features.drop(columns=exclude_variables).iloc[1], matplotlib=True, show=True, contribution_threshold=0.08)

# water plot
explanation = shap.Explanation(filtered_shap_values[1],  base_values = explainer.expected_value, feature_names = X_features.drop(columns=exclude_variables).columns)
shap.plots.waterfall(explanation, max_display=10, show=True)