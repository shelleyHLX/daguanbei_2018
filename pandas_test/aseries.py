# coding: utf-8

# import pandas as pd
# import numpy as np

# s = pd.Series()  # Series([], dtype: float64)
# print(s)

# data = np.array(['a', 'b', 'c', 'd'])
# s = pd.Series(data)
# print(s)

# data = np.array(['a', 'b', 'c', 'd'])
# s = pd.Series(data, index=[100, 101, 102, 103])
# print(s)

# data = {'a': 0., 'b':1, 'c':2, 'd':3}
# s = pd.Series(data)
# print(s)

# data = {'a':0, 'b':1, 'c':2}
# s = pd.Series(data, index=['b', 'c', 'd', 'a'])
# print(s)

# s = pd.Series(5, index=[0, 1, 2, 3, 4])
# print(s)

# s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
# print(s[0])

# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# dates = pd.date_range('20180820', periods=7)
# # print(dates)
# print('--'*16)
# df = pd.DataFrame(np.random.randn(7, 4), index=dates,
#                   columns=list('abcd'))
# print(df)

# df2 = pd.DataFrame({'a':1,
#                     'b':pd.Timestamp('20180820'),
#                     'c':pd.Series(1, index=list(range(4)),
#                                   dtype='float32'),
#                     'd':np.array([3] * 4, dtype='int32')})
# print(df2)

# dates = pd.date_range('20180820', periods=8)
# # print(dates)
# df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=list('abcd'))
# print(df.head())
# print(df.tail())

# dates = pd.date_range('20180820', periods=6)
# df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('abcd'))
# print(df.sort_values(by='b'))
# print(df['a'])

# import numpy as np
# import pandas as pd
# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# import pandas as pd
# import numpy as np

# dates = pd.date_range('20170101', periods=7)
# df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
# print('head----'*10)
# print(df.head())
# print('tail----'*10)
# print(df.tail())
# print('index----'*10)
# print(df.index)
# print('columns----'*10)
# print(df.columns)
# print('values----'*10)
# print(df.values)

# import pandas as pd
# import numpy as np
#
# # Create a Dictionary of series
# d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Minsu','Jack',
#    'Lee','David','Gasper','Betina','Andres']),
#    'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}
#
# # Create a DataFrame
# df = pd.DataFrame(d)
# print(df)
#
# # print(df.sum(1))
# print(df.mean())

# import pandas as pd
# import numpy as np
#
# N=20
#
# df = pd.DataFrame({
#     'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
#     'x': np.linspace(0,stop=N-1,num=N),
#     'y': np.random.rand(N),
#     'C': np.random.choice(['Low','Medium','High'],N).tolist(),
#     'D': np.random.normal(100, 10, size=(N)).tolist()
#     })
#
# for col in df.iterrows():
#     print(col)

# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
#
# print(df)
#
# for row_index,row in df.iterrows():
#    print (row_index,row)
