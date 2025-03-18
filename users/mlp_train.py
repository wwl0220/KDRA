import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder
import torch
import torch.optim as optim
import torch.nn as nn

from users.mlp_model import MLP_Model

attack_set = ['androiddefender', 'androidspy', 'avforandroid', 'avpass', 'beanbot', 'biige', 'charger', 'dowgin',
              'ewind', 'fakeapp', 'fakeappal', 'fakeav', 'fakeinst', 'fakejoboffer', 'fakemart', 'fakenotify',
              'faketaobao', 'feiwo', 'gooligan', 'jifake', 'jisut', 'kemoge', 'koler', 'koodous', 'lockerpin',
              'mazarbot', 'mobidash', 'nandrobox', 'penetho', 'plankton', 'pletor', 'porndroid', 'ransombo',
              'selfmite', 'shuanet', 'simplocker', 'smssniffer', 'svpeng', 'virusshield', 'wannalocker', 'youmi',
              'zsone']


col_set = {'androiddefender': ['Idle Mean', 'PSH Flag Count', 'ACK Flag Count', 'RST Flag Count', 'Fwd PSH Flags',
                                'Idle Max', 'Flow Duration', 'FIN Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count',
                               'Label'],
            'androidspy': ['Bwd Init Win Bytes', 'URG Flag Count', 'FIN Flag Count', 'Flow Duration', 'Flow IAT Max',
                           'PSH Flag Count', 'Label'],
            'avforandroid': ['Flow Duration', 'PSH Flag Count', 'RST Flag Count', 'Bwd Init Win Bytes', 'Flow IAT Max',
                             'FIN Flag Count', 'URG Flag Count', 'ACK Flag Count', 'Label'],
            'avpass': ['Flow IAT Max', 'PSH Flag Count', 'Bwd Init Win Bytes', 'ACK Flag Count', 'URG Flag Count',
                       'FIN Flag Count', 'RST Flag Count', 'Flow Duration', 'Label'],
            'beanbot': ['Fwd PSH Flags', 'FIN Flag Count', 'Fwd IAT Total', 'PSH Flag Count', 'ACK Flag Count',
                        'Bwd Init Win Bytes', 'Flow IAT Max', 'URG Flag Count', 'RST Flag Count', 'Flow Duration',
                        'FWD Init Win Bytes', 'Label'],
            'biige': ['Flow IAT Max', 'Flow Duration', 'Fwd IAT Total', 'FWD Init Win Bytes', 'RST Flag Count',
                      'Bwd Init Win Bytes', 'PSH Flag Count', 'URG Flag Count', 'FIN Flag Count', 'Label'],
            'charger': ['Bwd Packet Length Mean', 'FIN Flag Count', 'Bwd Segment Size Avg', 'Bwd Packet Length Min',
                        'Flow Duration', 'Bwd Init Win Bytes', 'Fwd Packet Length Min', 'URG Flag Count',
                        'PSH Flag Count', 'ACK Flag Count', 'RST Flag Count', 'FWD Init Win Bytes', 'Label'],
            'dowgin': ['RST Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'PSH Flag Count', 'Flow Duration',
                       'URG Flag Count', 'Bwd Init Win Bytes', 'Label'],
            'ewind': ['ACK Flag Count', 'Flow IAT Max', 'URG Flag Count', 'FIN Flag Count', 'RST Flag Count',
                      'Flow Duration', 'Bwd Init Win Bytes', 'PSH Flag Count', 'Label'],
            'fakeapp': ['URG Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'PSH Flag Count', 'Flow Duration',
                        'Bwd Init Win Bytes', 'RST Flag Count', 'FIN Flag Count', 'Label'],
            'fakeappal': ['Flow IAT Max', 'URG Flag Count', 'Bwd Init Win Bytes', 'Flow Duration', 'FIN Flag Count',
                          'ACK Flag Count', 'RST Flag Count', 'PSH Flag Count', 'Label'],
            'fakeav': ['PSH Flag Count', 'Fwd Seg Size Min', 'URG Flag Count', 'Flow IAT Max', 'FIN Flag Count',
                       'Flow Duration', 'Bwd Init Win Bytes', 'RST Flag Count', 'Label'],
            'fakeinst': ['Fwd IAT Total', 'URG Flag Count', 'Fwd IAT Max', 'FIN Flag Count', 'Flow IAT Max',
                         'Flow Duration', 'FWD Init Win Bytes', 'Fwd PSH Flags', 'PSH Flag Count', 'Label'],
            'fakejoboffer': ['Bwd Init Win Bytes', 'FWD Init Win Bytes', 'FIN Flag Count', 'Flow Duration',
                             'PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Label'],
            'fakemart': ['Fwd IAT Total', 'Bwd Init Win Bytes', 'RST Flag Count', 'PSH Flag Count', 'Fwd IAT Max',
                         'FWD Init Win Bytes', 'URG Flag Count', 'Flow Duration', 'FIN Flag Count', 'Flow IAT Max',
                         'Label'],
            'fakenotify': ['Fwd PSH Flags', 'PSH Flag Count', 'URG Flag Count', 'FIN Flag Count', 'Flow IAT Max',
                           'Fwd IAT Total', 'Flow Duration', 'Label'],
            'faketaobao': ['ACK Flag Count', 'Flow IAT Max', 'RST Flag Count', 'PSH Flag Count', 'Bwd Init Win Bytes',
                           'Flow Duration', 'FIN Flag Count', 'URG Flag Count', 'Label'],
            'feiwo': ['Flow Duration', 'FIN Flag Count', 'PSH Flag Count', 'Flow IAT Max', 'Bwd Init Win Bytes',
                      'ACK Flag Count', 'URG Flag Count', 'RST Flag Count', 'Label'],
            'gooligan': ['Packet Length Std', 'PSH Flag Count', 'Fwd IAT Total', 'FIN Flag Count', 'URG Flag Count',
                         'RST Flag Count', 'ACK Flag Count', 'FWD Init Win Bytes', 'Packet Length Max',
                         'Bwd Init Win Bytes', 'Packet Length Variance', 'Packet Length Mean', 'Flow Duration',
                         'Label'],
            'jifake': ['FWD Init Win Bytes', 'RST Flag Count', 'Flow Duration', 'FIN Flag Count', 'Fwd IAT Max',
                       'Fwd IAT Total', 'Bwd Init Win Bytes', 'Flow IAT Max', 'PSH Flag Count', 'URG Flag Count',
                       'Label'],
            'jisut': ['PSH Flag Count', 'FIN Flag Count', 'Bwd Init Win Bytes', 'Flow Duration', 'Fwd Packet Length Min',
                      'ACK Flag Count', 'URG Flag Count', 'RST Flag Count', 'Fwd PSH Flags', 'Label'],
            'kemoge': ['Fwd Seg Size Min', 'PSH Flag Count', 'FIN Flag Count', 'URG Flag Count', 'Packet Length Variance',
                       'Packet Length Std', 'RST Flag Count', 'Bwd Segment Size Avg', 'Flow Duration', 'Bwd Packet Length Mean',
                       'ACK Flag Count', 'Bwd Init Win Bytes', 'Label'],
            'koler': ['PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'Bwd Init Win Bytes', 'Flow Duration',
                      'FIN Flag Count', 'Fwd Packets/s', 'FWD Init Win Bytes', 'ACK Flag Count', 'Label'],
            'koodous': ['Fwd Seg Size Min', 'Bwd Segment Size Avg', 'Packet Length Std', 'Packet Length Max',
                        'PSH Flag Count', 'FIN Flag Count', 'Packet Length Variance', 'Bwd Packet Length Mean',
                        'Flow Duration', 'URG Flag Count', 'ACK Flag Count', 'Bwd Init Win Bytes', 'RST Flag Count',
                        'Label'],
            'lockerpin': ['FIN Flag Count', 'Bwd Init Win Bytes', 'PSH Flag Count', 'Fwd PSH Flags', 'URG Flag Count',
                          'Flow Duration', 'Label'],
            'mazarbot': ['PSH Flag Count', 'ACK Flag Count', 'Flow Duration', 'FIN Flag Count', 'URG Flag Count',
                         'Fwd PSH Flags', 'RST Flag Count', 'Flow IAT Max', 'Bwd Init Win Bytes', 'Label'],
            'mobidash': ['RST Flag Count', 'FIN Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Bwd Init Win Bytes',
                         'Flow Duration', 'PSH Flag Count', 'Label'],
            'nandrobox': ['Flow IAT Max', 'PSH Flag Count', 'RST Flag Count', 'URG Flag Count', 'Bwd Init Win Bytes',
                          'ACK Flag Count', 'Flow Duration', 'Fwd PSH Flags', 'FIN Flag Count', 'Label'],
            'penetho': ['URG Flag Count', 'Flow IAT Max', 'Fwd PSH Flags', 'PSH Flag Count', 'FIN Flag Count',
                        'Flow Duration', 'Label'],
            'plankton': ['Flow Duration', 'PSH Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count', 'FIN Flag Count',
                         'RST Flag Count', 'Bwd Segment Size Avg', 'Bwd Packet Length Mean', 'ACK Flag Count',
                         'Fwd PSH Flags', 'FWD Init Win Bytes', 'Label'],
            'pletor': ['Fwd IAT Total', 'Fwd Packet Length Min', 'FIN Flag Count', 'ACK Flag Count', 'Fwd PSH Flags',
                       'Idle Max', 'Flow IAT Max', 'Packet Length Std', 'Bwd Init Win Bytes', 'Flow Duration',
                       'URG Flag Count', 'RST Flag Count', 'PSH Flag Count', 'Fwd IAT Max', 'Label'],
            'porndroid': ['ACK Flag Count', 'FIN Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count', 'PSH Flag Count',
                          'FWD Init Win Bytes', 'Flow Duration', 'RST Flag Count', 'Label'],
            'ransombo': ['Fwd Packet Length Min', 'Bwd Packet Length Min', 'FIN Flag Count', 'ACK Flag Count',
                         'Fwd PSH Flags', 'Bwd Init Win Bytes', 'Flow IAT Max', 'URG Flag Count', 'Flow Duration',
                         'RST Flag Count', 'Packet Length Min', 'PSH Flag Count', 'Label'],
            'selfmite': ['URG Flag Count', 'PSH Flag Count', 'Flow IAT Max', 'Fwd PSH Flags', 'Bwd Packet Length Max',
                         'Flow Duration', 'FIN Flag Count', 'Label'],
            'shuanet': ['FIN Flag Count', 'Flow Duration', 'RST Flag Count', 'PSH Flag Count', 'URG Flag Count',
                        'Bwd Init Win Bytes', 'ACK Flag Count', 'Flow IAT Max', 'Label'],
            'simplocker': ['RST Flag Count', 'Bwd Segment Size Avg', 'Fwd Packet Length Min', 'Bwd Init Win Bytes',
                           'Fwd PSH Flags', 'FIN Flag Count', 'Bwd Packet Length Mean', 'PSH Flag Count', 'Flow IAT Max',
                           'URG Flag Count', 'Flow Duration', 'ACK Flag Count', 'FWD Init Win Bytes',
                           'Bwd Packet Length Min', 'Label'],
            'smssniffer': ['Idle Mean', 'Flow Duration', 'ACK Flag Count', 'FIN Flag Count', 'Bwd Segment Size Avg',
                           'Idle Max', 'RST Flag Count', 'URG Flag Count', 'Fwd PSH Flags', 'Bwd Packet Length Mean',
                           'PSH Flag Count', 'Bwd Init Win Bytes', 'Label'],
            'svpeng': ['Fwd Packet Length Min', 'ACK Flag Count', 'Flow Duration', 'Flow IAT Max', 'PSH Flag Count',
                       'FIN Flag Count', 'Fwd PSH Flags', 'RST Flag Count', 'Bwd Init Win Bytes', 'Idle Max',
                       'Packet Length Min', 'URG Flag Count', 'Label'],
            'virusshield': ['Flow Duration', 'PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'FIN Flag Count',
                            'Bwd Init Win Bytes', 'FWD Init Win Bytes', 'Flow IAT Max', 'ACK Flag Count',
                            'Fwd IAT Total', 'Label'],
            'wannalocker': ['Flow Duration', 'URG Flag Count', 'PSH Flag Count', 'Bwd Packet Length Min',
                            'RST Flag Count', 'Bwd Init Win Bytes', 'ACK Flag Count', 'Fwd Packet Length Min',
                            'Flow IAT Max', 'Idle Max', 'Packet Length Min', 'FIN Flag Count', 'Fwd PSH Flags', 'Label'],
            'youmi': ['ACK Flag Count', 'Flow Duration', 'RST Flag Count', 'URG Flag Count', 'FIN Flag Count',
                      'PSH Flag Count', 'Bwd Init Win Bytes', 'Label'],
            'zsone': ['Flow Duration', 'Packet Length Mean', 'Packet Length Std', 'PSH Flag Count', 'Fwd PSH Flags',
                      'Packet Length Max', 'FWD Init Win Bytes', 'RST Flag Count', 'FIN Flag Count', 'ACK Flag Count',
                      'Bwd Init Win Bytes', 'Packet Length Variance', 'URG Flag Count', 'Label']}

all_col = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
           'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets','Total Length of Fwd Packet',
           'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
           'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
           'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
           'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
           'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
           'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
           'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
           'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count',
           'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
           'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
           'Bwd Packet/Bulk Avg',  'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
           'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
           'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

n_layers = 3
hidden_dim = 16
output_dim = 2

max_epochs = 40
batch_size = 5000

torch.manual_seed(42)

for attack_name in attack_set:
    col = col_set[attack_name]
    input_dim = len(col) - 1

    file1 = r'../user_data/base_train_data/' + attack_name + '.csv'
    df1_1 = pd.read_csv(file1, low_memory=False, index_col=None, header=0)
    df1 = df1_1[col]
    df1 = df1.replace(np.nan, 0)
    df1 = df1.replace(np.inf, 0)
    attack_train_rows = df1.shape[0]

    file0 = r'../user_data/base_train_data/benign.csv'
    df0_0 = pd.read_csv(file0, low_memory=False, index_col=None, header=0, nrows=attack_train_rows)
    df0 = df0_0[col]
    df0 = df0.replace(np.nan, 0)
    df0 = df0.replace(np.inf, 0)

    file11 = r'../user_data/base_test_data/' + attack_name + '.csv'
    df11_11 = pd.read_csv(file11, low_memory=False, index_col=None, header=0)
    df11 = df11_11[col]
    df11 = df11.replace(np.nan, 0)
    df11 = df11.replace(np.inf, 0)
    attack_test_rows = df11.shape[0]

    file00 = r'../user_data/base_test_data/benign.csv'
    df00_00 = pd.read_csv(file00, low_memory=False, index_col=None, header=0, nrows=attack_test_rows)
    df00 = df00_00[col]
    df00 = df00.replace(np.nan, 0)
    df00 = df00.replace(np.inf, 0)

    df = pd.concat([df0, df1])
    data = df.values

    df_test = pd.concat([df00, df11])
    data_test = df_test.values

    X = data[:, 0: -1]
    Y = data[:, -1]
    print(Y.shape)
    Y_de_index = []
    for i in range(len(Y)):
        if Y[i] == 'Benign':
            Y[i] = 0
        elif Y[i] == 'SCAREWARE' or Y[i] == 'BENIGN' or Y[i] == 0:
            Y_de_index.append(i)
        else:
            Y[i] = 1
    X = np.delete(X, Y_de_index, axis=0)
    Y = np.delete(Y, Y_de_index)
    print(Y.shape)

    X_test = data_test[:, 0: -1]
    Y_test = data_test[:, -1]
    Y_test_de_index = []
    for i in range(len(Y_test)):
        if Y_test[i] == 'Benign':
            Y_test[i] = 0
        elif Y_test[i] == 'SCAREWARE' or Y_test[i] == 'BENIGN' or Y_test[i] == 0:
            Y_test_de_index.append(i)
        else:
            Y_test[i] = 1
    X_test = np.delete(X_test, Y_test_de_index, axis=0)
    Y_test = np.delete(Y_test, Y_test_de_index)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_min = scaler.data_min_
    X_max = scaler.data_max_
    print(X_min)
    X = scaler.transform(X)
    transformer = Normalizer().fit(X)
    X = transformer.transform(X)

    file_min = 'models/' + attack_name + 'X_min.pkl'
    file_max = 'models/' + attack_name + 'X_max.pkl'

    with open(file_min, 'wb') as f:
        pickle.dump(X_min, f)
    with open(file_max, 'wb') as f:
        pickle.dump(X_max, f)

    X_test = (X_test - X_min) / (X_max - X_min)
    transformer = Normalizer().fit(X_test)
    X_test = transformer.transform(X_test)

    encoder = OneHotEncoder(sparse=False)
    Y = np.array(Y).reshape(-1, 1)
    Y = encoder.fit_transform(Y)

    encoder = OneHotEncoder(sparse=False)
    Y_test = np.array(Y_test).reshape(-1, 1)
    Y_test = encoder.fit_transform(Y_test)

    X_train = torch.Tensor(X)
    Y_train = torch.Tensor(Y)
    X_test = torch.Tensor(X_test)
    Y_test = torch.Tensor(Y_test)

    mlp_model = MLP_Model(n_layers, input_dim, hidden_dim, output_dim)

    optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

    loss_func = nn.CrossEntropyLoss()

    mlp_model.train()
    epoch = 0
    while epoch < max_epochs:
        for i in range(0, len(X_train), batch_size):
            batch_data = X_train[i: i + batch_size]
            output = mlp_model(batch_data)
            batch_label = torch.argmax(Y_train[i: i + batch_size], dim=1)
            loss = loss_func(output, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, max_epochs, loss.item()))
        epoch += 1

    test_label = torch.argmax(Y_test, dim=1)

    mlp_model.eval()
    with torch.no_grad():
        # test_output = F.softmax(mlp_model(X_test), dim=1)
        test_output = mlp_model.predict_proba(X_test)
    test_predict = torch.argmax(test_output, dim=1)
    print(np.unique(test_predict))
    print(np.unique(test_label))

    test_acc = (test_predict == test_label).sum().item() / len(test_label)
    test_recall = ((test_predict == test_label) & (test_label == 1)).sum().item() / (test_label == 1).sum().item()
    test_fpr = ((test_predict != test_label) & (test_label == 0)).sum().item() / (test_label == 0).sum().item()
    print(test_acc)
    print(test_recall)
    print(test_fpr)

    model_path = 'models/' + attack_name + '.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(mlp_model, f)

    result_col = ['acc', 'recall', 'fpr']
    result = [[test_acc, test_recall, test_fpr]]
    df_result = pd.DataFrame(result, columns=result_col)
    df_result.to_csv('results/' + attack_name + '.csv', index=False)

    print('--------------------------------------')
