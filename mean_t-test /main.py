from scipy import stats
import numpy as np
import pandas as pd
import warnings

#行と列を省略しないオプション
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')# 警告を出ないようにする

def t_teat(data1, data2, one_to_one=True): 

  A = np.array(data1)
  B = np.array(data2)

  if one_to_one:
    print("<対応t検定>")
    print("帰無仮説 : 「対象の2群の平均値に差はない」について")
    answer = stats.ttest_rel(A, B)
    print(answer)
    if answer.pvalue < 0.05:
      print("と、p < 0.05と有意水準以下であるため、「2群間の平均値に差がない」の帰無仮説は棄却された。")
    else:
      print("と、p > 0.05と有意水準を満たしていないため、帰無仮説棄却ならず。")

  if not one_to_one:
    print("<非対応t検定>")
    A_var = np.var(A, ddof=1)  # Aの不偏分散
    B_var = np.var(B, ddof=1)  # Bの不偏分散
    A_df = len(A) - 1  # Aの自由度
    B_df = len(B) - 1  # Bの自由度

    f = A_var / B_var  # F比の値
    one_sided_pval1 = stats.f.cdf(f, A_df, B_df)  # 片側検定のp値 1
    one_sided_pval2 = stats.f.sf(f, A_df, B_df)   # 片側検定のp値 2
    two_sided_pval = min(one_sided_pval1, one_sided_pval2) * 2  # 両側検定のp値
    print("---ステップ1---\n帰無仮説 : 「対象の2群の分散に差はない」について")
    print('F:       ', round(f, 3))
    print('p-value: ', round(two_sided_pval, 4))
    if round(two_sided_pval, 4) < 0.05:
      print("と、p < 0.05と有意水準以下であるため、帰無仮説は棄却され、2群間は不等分散であることが示唆された。")
      print("\n---ステップ2---\n次に本題の帰無仮説 : 「2群間の平均値に差がない」についてWelchのt検定")
      answer = stats.ttest_ind(A, B, equal_var=False)
      print('p-value: ', round(answer.pvalue, 4))
      if answer.pvalue < 0.05:
        print("と、p < 0.05と有意水準以下であるため、「2群間の平均値に差がない」の対立仮説は棄却された。")
      else:
        print("と、p > 0.05と有意水準を満たしていないため、帰無仮説棄却ならず。")

    if round(two_sided_pval, 4) >= 0.05:
      print("と、p > 0.05なので、帰無仮説は棄却されず、2群間は等分散であること(少なくとも不等分散でないこと)が示唆された。")
      print("\n---ステップ2---\n次に本題の帰無仮説 : 「2群間の平均値に差がない」についてStudentのt検定")
      answer = stats.ttest_ind(A, B)
      print('p-value: ', round(answer.pvalue, 4))
      if answer.pvalue < 0.05:
        print("と、p < 0.05と有意水準以下であるため、「2群間の平均値に差がない」の帰無仮説は棄却された。")
      else:
        print("と、p > 0.05と有意水準を満たしていないため、帰無仮説棄却ならず。")