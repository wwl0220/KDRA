import pandas as pd
import pickle

from sklearn.preprocessing import Normalizer
import torch

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

attack_set = ['androiddefender', 'androidspy', 'avforandroid', 'avpass', 'beanbot', 'biige', 'charger', 'dowgin',
              'ewind', 'fakeapp', 'fakeappal', 'fakeav', 'fakeinst', 'fakejoboffer', 'fakemart', 'fakenotify',
              'faketaobao', 'feiwo', 'gooligan', 'jifake', 'jisut', 'kemoge', 'koler', 'koodous', 'lockerpin',
              'mazarbot', 'mobidash', 'nandrobox', 'penetho', 'plankton', 'pletor', 'porndroid', 'ransombo',
              'selfmite', 'shuanet', 'simplocker', 'smssniffer', 'svpeng', 'virusshield', 'wannalocker', 'youmi',
              'zsone']

traffic_set = ['androiddefender', 'androidspy', 'avforandroid', 'avpass', 'beanbot', 'biige', 'charger', 'dowgin',
              'ewind', 'fakeapp', 'fakeappal', 'fakeav', 'fakeinst', 'fakejoboffer', 'fakemart', 'fakenotify',
              'faketaobao', 'feiwo', 'gooligan', 'jifake', 'jisut', 'kemoge', 'koler', 'koodous', 'lockerpin',
              'mazarbot', 'mobidash', 'nandrobox', 'penetho', 'plankton', 'pletor', 'porndroid', 'ransombo',
              'selfmite', 'shuanet', 'simplocker', 'smssniffer', 'svpeng', 'virusshield', 'wannalocker', 'youmi',
              'zsone', 'benign']

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

prob_col = ['androiddefender_0', 'androiddefender_1', 'androidspy_0', 'androidspy_1', 'avforandroid_0', 'avforandroid_1',
            'avpass_0', 'avpass_1', 'beanbot_0', 'beanbot_1', 'biige_0', 'biige_1', 'charger_0', 'charger_1',
            'dowgin_0', 'dowgin_1', 'ewind_0', 'ewind_1', 'fakeapp_0', 'fakeapp_1', 'fakeappal_0', 'fakeappal_1',
            'fakeav_0', 'fakeav_1', 'fakeinst_0', 'fakeinst_1', 'fakejoboffer_0', 'fakejoboffer_1', 'fakemart_0', 'fakemart_1',
            'fakenotify_0', 'fakenotify_1', 'faketaobao_0', 'faketaobao_1', 'feiwo_0', 'feiwo_1', 'gooligan_0', 'gooligan_1',
            'jifake_0', 'jifake_1', 'jisut_0', 'jisut_1', 'kemoge_0', 'kemoge_1', 'koler_0', 'koler_1',
            'koodous_0', 'koodous_1', 'lockerpin_0', 'lockerpin_1', 'mazarbot_0', 'mazarbot_1', 'mobidash_0', 'mobidash_1',
            'nandrobox_0', 'nandrobox_1', 'penetho_0', 'penetho_1', 'plankton_0', 'plankton_1', 'pletor_0', 'pletor_1',
            'porndroid_0', 'porndroid_1', 'ransombo_0', 'ransombo_1', 'selfmite_0', 'selfmite_1', 'shuanet_0', 'shuanet_1',
            'simplocker_0', 'simplocker_1', 'smssniffer_0', 'smssniffer_1', 'svpeng_0', 'svpeng_1', 'virusshield_0',
            'virusshield_1', 'wannalocker_0', 'wannalocker_1', 'youmi_0', 'youmi_1', 'zsone_0', 'zsone_1']

for traffic in traffic_set:
    file_train = r'../user_data/meta_train_data/' + traffic + '_train.csv'
    train_df = pd.read_csv(file_train, low_memory=False, header=0, index_col=None)
    train_df.columns = col
    train_data_0 = torch.Tensor()

    file_val = r'../user_data/meta_train_data/' + traffic + '_val.csv'
    val_df = pd.read_csv(file_val, low_memory=False, header=0, index_col=None)
    val_df.columns = col
    val_data_0 = torch.Tensor()

    for attack_name in attack_set:
        use_fea = use_feas[attack_name]

        train_df_X = train_df[use_fea]
        train_data_X = train_df_X.values

        val_df_X = val_df[use_fea]
        val_data_X = val_df_X.values

        file_min = 'models/' + attack_name + 'X_min.pkl'
        file_max = 'models/' + attack_name + 'X_max.pkl'
        with open(file_min, 'rb') as f:
            X_min = pickle.load(f)
        with open(file_max, 'rb') as f:
            X_max = pickle.load(f)

        train_data_X = (train_data_X - X_min) / (X_max - X_min)
        transformer = Normalizer().fit(train_data_X)
        train_data_X = transformer.transform(train_data_X)
        train_data_X = torch.Tensor(train_data_X)

        val_data_X = (val_data_X - X_min) / (X_max - X_min)
        transformer = Normalizer().fit(val_data_X)
        val_data_X = transformer.transform(val_data_X)
        val_data_X = torch.Tensor(val_data_X)

        model_path = 'models/' + attack_name + '.pkl'
        with open(model_path, 'rb') as f:
            a_model = pickle.load(f)
        a_model.eval()

        train_predict = a_model.predict_proba(train_data_X)
        train_data_0 = torch.cat((train_data_0, train_predict), dim=1)

        val_predict = a_model.predict_proba(val_data_X)
        val_data_0 = torch.cat((val_data_0, val_predict), dim=1)

    df_train_prob = pd.DataFrame(train_data_0.detach().numpy(), columns=prob_col)
    df_train_prob.to_csv('../base/prob_samples/'+traffic+'_train.csv', index=False)

    df_val_prob = pd.DataFrame(val_data_0.detach().numpy(), columns=prob_col)
    df_val_prob.to_csv('../base/prob_samples/'+traffic+'_val.csv', index=False)
