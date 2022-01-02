def data_counter(data):
  '''
  input: list or Series data
  output: sample size
  '''
  print("---------------------------")
  print(f"「1」: {list(data).count(1)}")
  print(f"「0」: {list(data).count(0)}")
  print(f"「nan」: {Series(data).isnull().sum()}")
  print(f"「inf」: {list(data).count(np.inf)}")
  print(f"over absolute「0」: {len([i for i in data if 0 < abs(i)])}")
  print("---------------------------")  
  print(f"total sample size: {len(data)}")
