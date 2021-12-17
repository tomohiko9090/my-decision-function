def data_cheacker(data):
  print("特徴的なサンプルサイズ")
  print("---------------------------")
  print(f"0: {list(data).count(0)}個")
  print(f"nan: {Series(data).isnull().sum()}個")
  print(f"inf: {list(data).count(np.inf)}個")
  print("---------------------------")  
