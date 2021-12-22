def data_counter(data):
  '''
  input: list or Series data
  output: sample size
  '''
  print(f"total sample size: {len(data)}")
  print("---------------------------")
  print(f"absolute over「0」: {len([i for i in data if 0 < abs(i)])}")
  print(f"「0」: {list(data).count(0)}")
  print(f"「nan」: {Series(data).isnull().sum()}")
  print(f"「inf」: {list(data).count(np.inf)}")
  print("---------------------------")  
