# import yfinance as yf
# df = yf.download("BP.L", start="2015-01-01", end="2023-12-31")
# ##############
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# scaler = MinMaxScaler()
# data = scaler.fit_transform(df[['Close']])

# # Sequence generator
# def create_sequences(data, seq_len=60):
#     X, y = [], []
#     for i in range(seq_len, len(data)):
#         X.append(data[i-seq_len:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# X, y = create_sequences(data)
def prd():
    return 0
