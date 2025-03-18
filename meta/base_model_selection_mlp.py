import pandas as pd
import numpy as np
import random
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler, Normalizer


col = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
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

col_ids = ['Dst Port', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
           'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
           'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
           'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
           'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
           'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
           'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
           'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
           'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
           'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
           'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Header Length.1', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
           'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets',
           'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
           'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
           'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

attack_models = ['androiddefender', 'androidspy', 'avforandroid', 'avpass', 'beanbot', 'biige', 'charger',
                 'dowgin', 'ewind', 'fakeapp', 'fakeappal', 'fakeav', 'fakeinst', 'fakejoboffer', 'fakemart',
                 'fakenotify', 'faketaobao', 'feiwo', 'gooligan', 'jifake', 'jisut', 'kemoge', 'koler', 'koodous',
                 'lockerpin', 'mazarbot', 'mobidash', 'nandrobox', 'penetho', 'plankton', 'pletor', 'porndroid',
                 'ransombo', 'selfmite', 'shuanet', 'simplocker', 'smssniffer', 'svpeng', 'virusshield',
                 'wannalocker', 'youmi', 'zsone']

use_feas = {'androiddefender': ['Idle Mean', 'PSH Flag Count', 'ACK Flag Count', 'RST Flag Count', 'Fwd PSH Flags',
                                'Idle Max', 'Flow Duration', 'FIN Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count'],
            'androidspy': ['Bwd Init Win Bytes', 'URG Flag Count', 'FIN Flag Count', 'Flow Duration', 'Flow IAT Max',
                           'PSH Flag Count'],
            'avforandroid': ['Flow Duration', 'PSH Flag Count', 'RST Flag Count', 'Bwd Init Win Bytes', 'Flow IAT Max',
                             'FIN Flag Count', 'URG Flag Count', 'ACK Flag Count'],
            'avpass': ['Flow IAT Max', 'PSH Flag Count', 'Bwd Init Win Bytes', 'ACK Flag Count', 'URG Flag Count',
                       'FIN Flag Count', 'RST Flag Count', 'Flow Duration'],
            'beanbot': ['Fwd PSH Flags', 'FIN Flag Count', 'Fwd IAT Total', 'PSH Flag Count', 'ACK Flag Count',
                        'Bwd Init Win Bytes', 'Flow IAT Max', 'URG Flag Count', 'RST Flag Count', 'Flow Duration',
                        'FWD Init Win Bytes'],
            'biige': ['Flow IAT Max', 'Flow Duration', 'Fwd IAT Total', 'FWD Init Win Bytes', 'RST Flag Count',
                      'Bwd Init Win Bytes', 'PSH Flag Count', 'URG Flag Count', 'FIN Flag Count'],
            'charger': ['Bwd Packet Length Mean', 'FIN Flag Count', 'Bwd Segment Size Avg', 'Bwd Packet Length Min',
                        'Flow Duration', 'Bwd Init Win Bytes', 'Fwd Packet Length Min', 'URG Flag Count',
                        'PSH Flag Count', 'ACK Flag Count','RST Flag Count', 'FWD Init Win Bytes'],
            'dowgin': ['RST Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'PSH Flag Count', 'Flow Duration',
                       'URG Flag Count', 'Bwd Init Win Bytes'],
            'ewind': ['ACK Flag Count', 'Flow IAT Max', 'URG Flag Count', 'FIN Flag Count', 'RST Flag Count',
                      'Flow Duration', 'Bwd Init Win Bytes', 'PSH Flag Count'],
            'fakeapp': ['URG Flag Count', 'ACK Flag Count', 'Fwd Packet Length Min', 'PSH Flag Count', 'Flow Duration',
                        'Bwd Init Win Bytes', 'RST Flag Count', 'FIN Flag Count'],
            'fakeappal': ['Flow IAT Max', 'URG Flag Count', 'Bwd Init Win Bytes', 'Flow Duration', 'FIN Flag Count',
                          'ACK Flag Count', 'RST Flag Count', 'PSH Flag Count'],
            'fakeav': ['PSH Flag Count', 'Fwd Seg Size Min', 'URG Flag Count', 'Flow IAT Max', 'FIN Flag Count',
                       'Flow Duration', 'Bwd Init Win Bytes', 'RST Flag Count'],
            'fakeinst': ['Fwd IAT Total', 'URG Flag Count', 'Fwd IAT Max', 'FIN Flag Count', 'Flow IAT Max',
                         'Flow Duration', 'FWD Init Win Bytes', 'Fwd PSH Flags', 'PSH Flag Count'],
            'fakejoboffer': ['Bwd Init Win Bytes', 'FWD Init Win Bytes', 'FIN Flag Count', 'Flow Duration',
                             'PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'ACK Flag Count'],
            'fakemart': ['Fwd IAT Total', 'Bwd Init Win Bytes', 'RST Flag Count', 'PSH Flag Count', 'Fwd IAT Max',
                         'FWD Init Win Bytes', 'URG Flag Count', 'Flow Duration', 'FIN Flag Count', 'Flow IAT Max'],
            'fakenotify': ['Fwd PSH Flags', 'PSH Flag Count', 'URG Flag Count', 'FIN Flag Count', 'Flow IAT Max',
                           'Fwd IAT Total', 'Flow Duration'],
            'faketaobao': ['ACK Flag Count', 'Flow IAT Max', 'RST Flag Count', 'PSH Flag Count', 'Bwd Init Win Bytes',
                           'Flow Duration', 'FIN Flag Count', 'URG Flag Count'],
            'feiwo': ['Flow Duration', 'FIN Flag Count', 'PSH Flag Count', 'Flow IAT Max', 'Bwd Init Win Bytes',
                      'ACK Flag Count', 'URG Flag Count', 'RST Flag Count'],
            'gooligan': ['Packet Length Std', 'PSH Flag Count', 'Fwd IAT Total', 'FIN Flag Count', 'URG Flag Count',
                         'RST Flag Count', 'ACK Flag Count', 'FWD Init Win Bytes', 'Packet Length Max',
                         'Bwd Init Win Bytes', 'Packet Length Variance', 'Packet Length Mean', 'Flow Duration'],
            'jifake': ['FWD Init Win Bytes', 'RST Flag Count', 'Flow Duration', 'FIN Flag Count', 'Fwd IAT Max',
                       'Fwd IAT Total', 'Bwd Init Win Bytes', 'Flow IAT Max', 'PSH Flag Count', 'URG Flag Count'],
            'jisut': ['PSH Flag Count', 'FIN Flag Count', 'Bwd Init Win Bytes', 'Flow Duration', 'Fwd Packet Length Min',
                      'ACK Flag Count', 'URG Flag Count', 'RST Flag Count', 'Fwd PSH Flags'],
            'kemoge': ['Fwd Seg Size Min', 'PSH Flag Count', 'FIN Flag Count', 'URG Flag Count', 'Packet Length Variance',
                       'Packet Length Std', 'RST Flag Count', 'Bwd Segment Size Avg', 'Flow Duration', 'Bwd Packet Length Mean',
                       'ACK Flag Count', 'Bwd Init Win Bytes'],
            'koler': ['PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'Bwd Init Win Bytes', 'Flow Duration',
                      'FIN Flag Count', 'Fwd Packets/s', 'FWD Init Win Bytes', 'ACK Flag Count'],
            'koodous': ['Fwd Seg Size Min', 'Bwd Segment Size Avg', 'Packet Length Std', 'Packet Length Max',
                        'PSH Flag Count', 'FIN Flag Count', 'Packet Length Variance', 'Bwd Packet Length Mean',
                        'Flow Duration', 'URG Flag Count', 'ACK Flag Count', 'Bwd Init Win Bytes', 'RST Flag Count'],
            'lockerpin': ['FIN Flag Count', 'Bwd Init Win Bytes', 'PSH Flag Count', 'Fwd PSH Flags', 'URG Flag Count',
                          'Flow Duration'],
            'mazarbot': ['PSH Flag Count', 'ACK Flag Count', 'Flow Duration', 'FIN Flag Count', 'URG Flag Count',
                         'Fwd PSH Flags', 'RST Flag Count', 'Flow IAT Max', 'Bwd Init Win Bytes'],
            'mobidash': ['RST Flag Count', 'FIN Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Bwd Init Win Bytes',
                         'Flow Duration', 'PSH Flag Count'],
            'nandrobox': ['Flow IAT Max', 'PSH Flag Count', 'RST Flag Count', 'URG Flag Count', 'Bwd Init Win Bytes',
                          'ACK Flag Count', 'Flow Duration', 'Fwd PSH Flags', 'FIN Flag Count'],
            'penetho': ['URG Flag Count', 'Flow IAT Max', 'Fwd PSH Flags', 'PSH Flag Count', 'FIN Flag Count',
                        'Flow Duration'],
            'plankton': ['Flow Duration', 'PSH Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count', 'FIN Flag Count',
                         'RST Flag Count', 'Bwd Segment Size Avg', 'Bwd Packet Length Mean', 'ACK Flag Count',
                         'Fwd PSH Flags', 'FWD Init Win Bytes'],
            'pletor': ['Fwd IAT Total', 'Fwd Packet Length Min', 'FIN Flag Count', 'ACK Flag Count', 'Fwd PSH Flags',
                       'Idle Max', 'Flow IAT Max', 'Packet Length Std', 'Bwd Init Win Bytes', 'Flow Duration',
                       'URG Flag Count', 'RST Flag Count', 'PSH Flag Count', 'Fwd IAT Max'],
            'porndroid': ['ACK Flag Count', 'FIN Flag Count', 'Bwd Init Win Bytes', 'URG Flag Count', 'PSH Flag Count',
                          'FWD Init Win Bytes', 'Flow Duration', 'RST Flag Count'],
            'ransombo': ['Fwd Packet Length Min', 'Bwd Packet Length Min', 'FIN Flag Count', 'ACK Flag Count',
                         'Fwd PSH Flags', 'Bwd Init Win Bytes', 'Flow IAT Max', 'URG Flag Count', 'Flow Duration',
                         'RST Flag Count', 'Packet Length Min', 'PSH Flag Count'],
            'selfmite': ['URG Flag Count', 'PSH Flag Count', 'Flow IAT Max', 'Fwd PSH Flags', 'Bwd Packet Length Max',
                         'Flow Duration', 'FIN Flag Count'],
            'shuanet': ['FIN Flag Count', 'Flow Duration', 'RST Flag Count', 'PSH Flag Count', 'URG Flag Count',
                        'Bwd Init Win Bytes', 'ACK Flag Count', 'Flow IAT Max'],
            'simplocker': ['RST Flag Count', 'Bwd Segment Size Avg', 'Fwd Packet Length Min', 'Bwd Init Win Bytes',
                           'Fwd PSH Flags', 'FIN Flag Count', 'Bwd Packet Length Mean', 'PSH Flag Count', 'Flow IAT Max',
                           'URG Flag Count', 'Flow Duration', 'ACK Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min'],
            'smssniffer': ['Idle Mean', 'Flow Duration', 'ACK Flag Count', 'FIN Flag Count', 'Bwd Segment Size Avg',
                           'Idle Max', 'RST Flag Count', 'URG Flag Count', 'Fwd PSH Flags', 'Bwd Packet Length Mean',
                           'PSH Flag Count', 'Bwd Init Win Bytes'],
            'svpeng': ['Fwd Packet Length Min', 'ACK Flag Count', 'Flow Duration', 'Flow IAT Max', 'PSH Flag Count',
                       'FIN Flag Count', 'Fwd PSH Flags', 'RST Flag Count', 'Bwd Init Win Bytes', 'Idle Max',
                       'Packet Length Min', 'URG Flag Count'],
            'virusshield': ['Flow Duration', 'PSH Flag Count', 'URG Flag Count', 'RST Flag Count', 'FIN Flag Count',
                            'Bwd Init Win Bytes', 'FWD Init Win Bytes', 'Flow IAT Max', 'ACK Flag Count', 'Fwd IAT Total'],
            'wannalocker': ['Flow Duration', 'URG Flag Count', 'PSH Flag Count', 'Bwd Packet Length Min',
                            'RST Flag Count', 'Bwd Init Win Bytes', 'ACK Flag Count', 'Fwd Packet Length Min',
                            'Flow IAT Max', 'Idle Max', 'Packet Length Min', 'FIN Flag Count', 'Fwd PSH Flags'],
            'youmi': ['ACK Flag Count', 'Flow Duration', 'RST Flag Count', 'URG Flag Count', 'FIN Flag Count',
                      'PSH Flag Count', 'Bwd Init Win Bytes'],
            'zsone': ['Flow Duration', 'Packet Length Mean', 'Packet Length Std', 'PSH Flag Count', 'Fwd PSH Flags',
                      'Packet Length Max', 'FWD Init Win Bytes', 'RST Flag Count', 'FIN Flag Count', 'ACK Flag Count',
                      'Bwd Init Win Bytes', 'Packet Length Variance', 'URG Flag Count']}


def select_base_model(k, new_attack, data_shot, seeds):
    file_train_attack = r'../../new_attack_data/' + new_attack + '_train.csv'
    file_val_attack = r'../../new_attack_data/' + new_attack + '_val.csv'

    random.seed(seeds)
    skip_rows = random.randint(0, 49)

    df_train_attack = pd.read_csv(file_train_attack, low_memory=False, skiprows=skip_rows, nrows=data_shot * 2, header=0, index_col=None)
    df_train_attack.columns = col_ids  # ftp/hulk col_ids
    df_train_attack = df_train_attack.replace(np.nan, 0)
    df_train_attack = df_train_attack.replace(np.inf, 0)
    df_val_attack = pd.read_csv(file_val_attack, low_memory=False, skiprows=skip_rows, nrows=data_shot * 2, header=0, index_col=None)
    df_val_attack.columns = col_ids
    df_val_attack = df_val_attack.replace(np.nan, 0)
    df_val_attack = df_val_attack.replace(np.inf, 0)

    df_attack = pd.concat([df_train_attack, df_val_attack])

    scores = {}
    for attack_name in attack_models:
        use_fea = use_feas[attack_name]

        df_X = df_attack[use_fea]
        X = df_X.values

        file_min = '../../base/mlp/models/' + attack_name + 'X_min.pkl'
        file_max = '../../base/mlp/models/' + attack_name + 'X_max.pkl'
        with open(file_min, 'rb') as f:
            X_min = pickle.load(f)
        with open(file_max, 'rb') as f:
            X_max = pickle.load(f)

        X = (X - X_min) / (X_max - X_min)
        transformer = Normalizer().fit(X)
        X = transformer.transform(X)
        X = torch.Tensor(X)

        model_path = '../../base/mlp/models/' + attack_name + '.pkl'
        with open(model_path, 'rb') as f:
            a_model = pickle.load(f)
        a_model.eval()

        pred = a_model.predict_proba(X)
        pred = pred.detach().numpy()

        score = 0
        for i in range(len(pred)):
            score_i = pred[i][1] - pred[i][0]
            score += score_i
        scores[attack_name] = score / len(pred)

    print(scores)
    sorted_models = sorted(scores, key=scores.get, reverse=True)
    selected_models = sorted_models[0: k]
    print(selected_models)

    sorted_scores = []
    for i in sorted_models:
        sorted_scores.append(scores[i])
    print(sorted_scores)

    return selected_models

# select_base_model(5, 'cc', 10, 42)
