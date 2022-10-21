from curses import raw
from matplotlib.pyplot import axis
import numpy as np
import sklearn.preprocessing as sp

class preprocessor():
    def data_standardlize(self, procdata):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(procdata)

    def data_absmax_scale(self, procdata):
        return sp.maxabs_scale(procdata, axis=1)

    def data_minmax_scale(self, procdata):
        return sp.minmax_scale(procdata, axis=1)

    def data_normalizer(self, procdata):
        return sp.normalize(procdata, norm='l2', axis=1)

    def data_reshape(self, reshape_cnn=False, reshape_data=[]):
        if type(reshape_data) != np.ndarray:
            raise TypeError('傳入資料非numpy array! Not numpy Array!')
        if reshape_cnn == '1d':
            return np.reshape(reshape_data, (reshape_data.shape[0], 1, reshape_data.shape[1]))
        if reshape_cnn == '2d':
            temp = reshape_data.copy()
            temp.resize((temp.shape[0], 9, 9), refcheck=False)
            return np.reshape(temp, (temp.shape[0], 1, temp.shape[1], temp.shape[2]))
        return np.reshape(reshape_data, (reshape_data.shape[0], reshape_data.shape[1]))

    def __init__(self) -> None:
        self.felist = ['Flow Duration', 'Total Fwd Packet',
       'Total Bwd packets', 'Total Length of Fwd Packet',
       'Total Length of Bwd Packet', 'Fwd Packet Length Max',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Max',
       'Bwd Packet Length Min', 'Bwd Packet Length Mean',
       'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
       'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
       'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
       'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
       # 'Fwd RST Flags', 'Bwd RST Flags',
       'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
       'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
       'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
       'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
       'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
       'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
       'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
       'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
       'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
       'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
       'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min']
        self.droplst = ['Label']
        #, 'DoS GoldenEye - Attempted', 'DoS Hulk - Attempted', 'DoS Slowhttptest - Attempted', 'DoS slowloris - Attempted']
        self.col_list={
            'M1':['Flow ID', 'Src IP', 'Src Port',
        'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
        'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
        'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
        'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean',
        'Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min',
        'Bwd Packet Length Mean','Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min','Fwd IAT Total',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max' ,'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
        'Average Packet Size', 'Fwd Segment Size Avg',
        'Bwd Segment Size Avg', 'Fwd Header Length.1',
        'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
        'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'],
            'M2019':['Unnamed: 0', 'Flow ID', 'Src IP', 'Src Port',
        'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
        'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
        'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
        'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean',
        'Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min',
        'Bwd Packet Length Mean','Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min','Fwd IAT Total',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max' ,'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
        'Average Packet Size', 'Fwd Segment Size Avg',
        'Bwd Segment Size Avg', 'Fwd Header Length.1',
        'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
        'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'SimillarHTTP', 'Inbound', 'Label']
        }



    def data_preprocess(self, dataset_path='', mn='M0', wbo=True, filter_l=None, ac=False, verbose=None):
        self.sort_lst = ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'Heartbleed']
        if type(ac) == list:
            self.sort_lst = self.sort_lst + ac
        import os.path as path
        if not path.exists(dataset_path):
            raise ValueError(f'{dataset_path} 找不到資料集. Dataset files not found.')
        import pandas as pd
        pd.options.display.max_rows = None
        rawdf = pd.read_csv(dataset_path, dtype='unicode')
        rawdf.rename(columns=lambda x: x.strip(), inplace=True)
        if verbose:
            print(rawdf.shape, rawdf.columns.shape)
        new_cols = self.col_list.get(mn)
        if new_cols != None:
            rawdf.columns = new_cols
        else:
            del new_cols
        # if dbn:#不保留良性
        if wbo == False:
            rawdf = rawdf[rawdf.Label != 'B_other']
        if type(filter_l) == list:
            rawdf = rawdf[rawdf.Label.isin(filter_l)]
        rawdf.replace(to_replace=['Infinity'], value='10000000000', inplace=True)
        rawdf.replace(to_replace=['NaN'], value=np.nan, inplace=True)
        rawdf.dropna(inplace=True)
        X = rawdf.get(self.felist).copy()
        if verbose:
            print(X.shape, X.columns)
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.astype('float32')
        X[X < 0] = 0
        X = np.log10(X + 1) / 10
        X = pd.concat([X, rawdf.Label], axis=1)
        # X = X.loc[:, (X != 0).any(axis=0)]
        Y = pd.get_dummies(X.Label)
        for i_ in self.sort_lst:
            if i_ not in Y.columns:
                Y[i_] = 0
        Y = Y[self.sort_lst]
        Y = Y.fillna(0)
        # X = pd.concat([rawdf.Protocol.astype('float32'), X], axis=1)
        # Y = X.Label
        X = X.drop(columns=self.droplst)
        final_cols = X.columns
        if verbose:
            print(X.min(), X.max())
        X = X.to_numpy()
        if verbose:
            print(final_cols)
        print(Y.value_counts())
        return X, Y, final_cols

    def data_preprocess_Bin(self, dataset_path='', mn='M0', wbo=True, filter_l=[], ac=False):
        self.sort_lst = ['BENIGN', 'MALICIOUS']
        import os.path as path
        if not path.exists(dataset_path):
            raise ValueError(f'{dataset_path} 找不到資料集. Dataset files not found.')
        import pandas as pd
        pd.options.display.max_rows = None
        rawdf = pd.read_csv(dataset_path, dtype='unicode')
        rawdf.rename(columns=lambda x: x.strip(), inplace=True)
        print(rawdf.shape, rawdf.columns.shape)
        new_cols = self.col_list.get(mn)
        if new_cols != None:
            rawdf.columns = new_cols
        else:
            del new_cols
        # if dbn:#不保留良性
        rawdf.Label[rawdf.Label == 'BENIGN'] = 'BENIGN'
        rawdf.Label[rawdf.Label != 'BENIGN'] = 'MALICIOUS'
        if filter_l.__len__() > 0:
            rawdf = rawdf[rawdf.Label.isin(filter_l)]
        print(rawdf.shape)
        rawdf.replace(to_replace=['Infinity'], value='10000000000', inplace=True)
        rawdf.replace(to_replace=['NaN'], value=np.nan, inplace=True)
        rawdf.dropna(inplace=True)
        X = rawdf.get(self.felist).copy()
        print(X.shape, X.columns)
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.astype('float32')
        X[X < 0] = 0
        X = np.log10(X + 1) / 10
        X = pd.concat([X, rawdf.Label], axis=1)
        # X = X.loc[:, (X != 0).any(axis=0)]
        Y = pd.get_dummies(X.Label)
        for i_ in self.sort_lst:
            if i_ not in Y.columns:
                Y[i_] = 0
        Y = Y[self.sort_lst]
        Y = Y.fillna(0)
        # X = pd.concat([rawdf.Protocol.astype('float32'), X], axis=1)
        # Y = X.Label
        X = X.drop(columns=self.droplst)
        final_cols = X.columns
        print(X.min(), X.max())
        X = X.to_numpy()
        print(final_cols, '\n', Y.value_counts())
        return X, Y, final_cols