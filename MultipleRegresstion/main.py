import numpy as np
import pandas as pd
from pandas import DataFrame 
import itertools, sklearn
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt

class MultipleRegresstion:

  def __init__(self, table, target, normalize=False):
    self.normalize = normalize
    self.table = table
    self.target = target

  def choice_regresstion(self, limit, test_size=False):

    print("--- 1. モデル ---")
    # 1. detasetの作成
    print(f"サンプルサイズ：{len(self.table)}") 
    
    X_multi = self.table.drop(self.target, 1)
    Y_target = self.table[self.target]

    if self.normalize == "standardization": # 普通の分散を使う
      X_multi = (X_multi - X_multi.mean()) / X_multi.std(ddof=0) 

    if self.normalize == "min-max-scaling": # 最大値が1, 最小値が0になるように正規化
      X_multi = X_multi.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
      Y_target = Y_target.apply(lambda y: (y - np.min(Y_target)) / (np.max(Y_target) - np.min(Y_target)))

    # 2.1. モデルを作成
    lreg = LinearRegression()
    lreg.fit(X_multi, Y_target)

    # 2.2. 自由度調整済み決定係数
    X1 = sm.add_constant(X_multi)
    m = sm.OLS(Y_target, X1)
    result = m.fit()

    # 3. 作成したモデルの回帰係数を確認
    print(f'切片の値: {lreg.intercept_:0.2f}')
    coeff_df = DataFrame({"Coefficient": ["β" + str(i+1) for i in range(len(X_multi.columns))]})
    coeff_df["Features"] = X_multi.columns
    coeff_df["Coefficient Estimate"] = pd.Series(lreg.coef_)
    corr_pearson_list = []
    p_value_list = []
    y = Y_target
    for column_name in X_multi:
      x = X_multi[column_name]
      corr_pearson, p_value = pearsonr(x, y)  
      corr_pearson_list.append(round(corr_pearson, 3)) 
      p_value_list.append(round(p_value, 3))
    coeff_df["corr pearson"] = corr_pearson_list
    coeff_df["p-value"] = p_value_list

    print(f"自由度調整済み決定係数: {round(result.rsquared_adj, 3)}")
    print(f"AIC: {round(result.aic, 2)}")
    print(coeff_df, "\n")
   
    print("--- 2. 予測精度 ---")
    print("パラメータの信頼性を判断する")

    loo = LeaveOneOut()

    X = X_multi.reset_index().values
    y = Y_target.reset_index().values
    lreg = LinearRegression()

    loo_result_list = []
    for train_index, test_index in loo.split(X):

      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      lreg = LinearRegression()
      lreg.fit(X_train, y_train)
      pred_test = lreg.predict(X_test)
      mae = mean_absolute_error(y_test, pred_test)
      loo_result_list.append(mae)

    print(f"\n1. LeaveOneOut（MAEの平均値）: {round(np.mean(loo_result_list), 3)}\n")

    if test_size:
      X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_multi, Y_target, test_size=test_size)

    if not test_size:
      X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_multi, Y_target)

    print(f"学習用のデータ数 -> {len(X_train)}")
    print(f"検証用のデータ数 -> {len(X_test)}")
    lreg = LinearRegression()
    lreg.fit(X_train, Y_train)

    pred_train = lreg.predict(X_train)
    pred_test = lreg.predict(X_test)
    
    print(f"\n2. MAE 平均絶対誤差")
    print(f"X_trainを使ったモデル: {mean_absolute_error(Y_train, pred_train):0.3f}")
    print(f"X_testを使ったモデル: {mean_absolute_error(Y_test, pred_test):0.3f}")

    print(f"\n3. MSE 平均二乗誤差")
    print(f"X_trainを使ったモデル: {mean_squared_error(Y_train, pred_train):0.3f}")
    print(f"X_testを使ったモデル: {mean_squared_error(Y_test, pred_test):0.3f}")
    
    print(f"\n4. RMSE 平均二乗誤差の平方根")
    print(f"X_trainを使ったモデル: {np.sqrt(mean_squared_error(Y_train, pred_train)):0.3f}")
    print(f"X_testを使ったモデル: {np.sqrt(mean_squared_error(Y_test, pred_test)):0.3f}")
    
    # 予測精度のプロット
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(Y_train, pred_train, label="Training", s=10, c='b',alpha=0.3)
    ax.scatter(Y_test, pred_test, label="Test", s=10, c='r',alpha=0.3)
    ax.plot([limit[0], limit[1]], [limit[0], limit[1]], c="black", alpha=0.8)
    ax.set_title('yyplot')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.legend(frameon=False, fontsize=8)

    gray = "#CDCCC9"
    ax.spines['left'].set_color(gray)
    ax.spines['right'].set_color(gray)
    ax.spines['top'].set_color(gray)
    ax.spines['bottom'].set_color(gray)
    ax.tick_params(left=True, labelsize=6)
    
  def search_features_regresstion(self, feature_name_list:list=False, avoid_overtraining:int=False, avoid_multicollinearity:bool=False):
    
    if feature_name_list:
      table = self.table.rename(columns=dict(zip(self.table.columns[1:], feature_name_list)))
    if not feature_name_list:
      table = self.table

    # 1. 全組み合わせを作成
    result = []
    feature_num = len(table.drop(self.target, 1).columns)
    if avoid_overtraining:
      feature_num = avoid_overtraining
    for n in range(1, feature_num+1):
      for conb in itertools.combinations(table.drop(self.target, 1).columns, n):
          result.append(list(conb))
   
    # 2. 作成する決定係数df
    all_result_df = DataFrame(columns = ["AIC, r2_score", "VIF_max"] + list(table.drop(self.target, 1).columns))

    # 3. 一個ずつregresstion 
    for choice_parameter_list in result:

      X_multi = table[choice_parameter_list]
      Y_target = table[self.target]

      if self.normalize == "standardization": # 普通の分散を使う
        X_multi = (X_multi - X_multi.mean()) / X_multi.std(ddof=0) 

      if self.normalize == "min-max-scaling": # 最大値が1, 最小値が0になるように正規化
        X_multi = X_multi.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        Y_target = Y_target.apply(lambda y: (y - np.min(Y_target)) / (np.max(Y_target) - np.min(Y_target)))

      # 4.1 モデルを作成
      lreg = LinearRegression()
      lreg.fit(X_multi, Y_target)

      # 4.2. 自由度調整済み決定係数
      X1 = sm.add_constant(X_multi)
      m = sm.OLS(Y_target, X1)
      result = m.fit()

      # 5. VIFが一番高いものを
      corr_mat = np.array(X_multi.corr())
      inv_corr_mat = np.linalg.pinv(corr_mat) # 擬似逆行列 (pseudo-inverse matrix) 
      vif_max = np.max(np.diag(inv_corr_mat))

      # 6. 結果をデータフレームに入れる
      columns = ["AIC", "r2_score", "VIF_max"] + choice_parameter_list
      values = [round(result.aic, 4), round(result.rsquared_adj, 3), vif_max] + list(lreg.coef_)
      result_df = DataFrame([values], columns = columns)

      all_result_df = pd.concat([result_df, all_result_df])

    print("<モデル比較表>")
    all_result_df = all_result_df.replace(np.nan, '')
    all_result_df = all_result_df.sort_values('AIC', ascending=True)
    all_result_df = all_result_df.drop(["AIC, r2_score"], 1)
    all_result_df = all_result_df.reindex(columns=["AIC", "r2_score", "VIF_max"]+list(table.drop(self.target, 1).columns))

    if avoid_multicollinearity:
      all_result_df = all_result_df[all_result_df.VIF_max < 10] # VIF_maxが10以上のモデルは削除
  
    # 7. 完成したデータフレームをreturn
    return all_result_df.reset_index(drop=True)
