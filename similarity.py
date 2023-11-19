import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
from utils import return_args, save_json

config = return_args()

data_examples = pd.read_csv("dataset/example2_nodiscrization_3.csv")
demos_data = {'bar': ['_Tyr:5', 'tompinder:13', 'futureboy:0', 'yoavweiss:0', 'ubun:12', 'alexisss:14', 'ayana.whiteheadsmith:2', 'rebeca.dioce:3', 'rjfleck:3', 'koushik.sbiiit:22', 'peluz:1', 'abhijitlondhe:22', 'ahmedsaab:12', 'airwickzoo:2', 'DavidJBianco:14', 'didiwai:247', 'sablanchard:2', 'thilina454d:37', 'babsndeep:0', 'VitasAnderson:1', 'gnedster:7', 'Jabawakai:45', 'jc041304:1', 'torikelly:1', 'knight00800:0', 'namkyu.park1:0', 'UladzislauVarabei:75', 'AlessiaT:4', 'devinp:7', 'hinkyeol.woo:10', 'matiya:3', 'rmirandarj:5', 'roadster11x:0', 'jevelezse:36', 'keithtxw:23', 'lucky11:4', 'FakePersona:2', 'hadyelsahar:0', 'javierramirez:4', 'cunyauthor:18'], 'line': ['david24v:18', 'mckinstryt:4', 'matanatr96:9', 'RT03:1', '202438:3', 'athbyron:11', 'kmthakur:6', 'Maks:1', 'bloudermilk:7', 'dougkelly:31', 'f.attila16:5', 'gwinkler:0', 'jtbthomas2:0', 'lukesmolo92:0', 'maxbatt:5', 'max_tit:0', 'olemunch:476', 'scubacarlo:9', 'sdfgiuihjiuqaiu:2', 'sohrabrs:47', 'Frederikph:16', 'hadro:9', 'ines.chami:126', 'ats2660:3', 'janisz:2', 'Skee09:1', 'Chopper1901:186', 'robertomenezes:4', 'ashishkatnoria:5', 'sangee:169', 'paul.dylan.clark:2', 'Servinal:1', 'sidgoyal1:1', 'test12345:219', 'ylkjick532428:9', 'antrmendoza3:12', 'elovell1:3', 'IsaacCastromayor:3', 'GustavoMamaoAgnish:8', 'jchiuRP:4'], 'scatter': ['alizauf:29', 'FatherMaldoon:6', 'jdu:52', 'kalpana74:79', 'KoenVerschuren:0', 'mrk-andreev:30', 'mszym:39', 'shawncaeiro:59', 'toegan:0', 'vedangw:6', 'weiyi:0', 'AlexDataScientist:38', 'dem0sh:35', 'espac1o:0', 'geet93:131', 'hari.sri.s:72', 'jbochi:11', 'justinm.hsi:2', 'kinson:6', 'maxyan:145', 'mingyangzhou:360', 'myan_berkeley:2', 'ramiroap1612:60', 'shijie.zhao.gt:298', 'Thanmai:17', 'tzs:157', 'a.elefsiniotis:0', 'aficionado:2', 'otkoge:0', 'acarlsson2:30', 'mehul.d:0', 'mzalaya:0', 'pat.vdleer:5', 'reuben.pereira126:48', 'snakedog:12', '204876:7', 'ChristopherEllis:12', 'Evan204826:15', 'jinglu921:1', 'sarvagyas12:6'], 'box': ['jdoen1990:1', 'jhadley:36', 'sruthiravula:3', 'a.gudakov:89', 'nmming90:10', 'elizabeth1086:32', 'hongxu1013041980:30', 'alicia.b:5', 'HeatherVanGerven:1', 'Geraldo:13', 'ryan_v:11', 'dustyneggers:72', 'JonatanFigueroaGil:10', 'guitchounts:20', 'Laura.Fosso:21', 'avasquez:1', 'CassK:2', 'andreatm14:3', 'leesum:3', 'ahc72:2', 'bengin:42', '20.issareeya.b:9', 'ed.kk.ho:1', 'lyh:32', 'Runningman2:1', 'KcronoZz:2', 'kunal703:18', 'Lamanc:0', 'julia.hughes:5', 'mizanharris:51', '19naikittia.brayman:63', 'aymanmt:10', 'exatoa:9', 'zgcarvalho:5', 'nickdstew:1', 'xtralucky13:1', 'brianna.holland:7', 'pamela920:89', 'hzclinger:6', 'peecee:31']}

demos_data_train = {}
demos_data_test = {}
for k, v in demos_data.items():
	random.seed(config['seed'])
	random.shuffle(v)
	if k not in demos_data_train:
		demos_data_train[k] = v[:15]
	if k not in demos_data_test:
		demos_data_test[k] = v[15:]

# print (demos_data_train)

demos_data_list = []

for k, v in demos_data.items():
    demos_data_list.extend(v)

demos_ids_list = []
for name in demos_data_list:
    df_ = data_examples.loc[data_examples['fid'] == name]
    id_ = df_.index[0]
    demos_ids_list.append(id_)

demos_data_train_list = []
for k, v in demos_data_train.items():
    demos_data_train_list.extend(v)

demos_data_test_list = []
for k, v in demos_data_test.items():
    demos_data_test_list.extend(v)

demos_id_data_dict = dict([(vv,kk) for kk, vv in enumerate(demos_data_list)])


demo_df = data_examples.iloc[demos_ids_list]
demo_list = demo_df.to_dict("records")

label_list = list(demo_df['trace_type'])
demos_names_list = list(demo_df['fid'])

name_label_dict = dict(zip(demos_names_list,label_list))

features = list(demo_df.columns)[2:-1]

demo_df_x = demo_df[features]
numerical_fea = list(demo_df_x.select_dtypes(exclude=['object']).columns)
demo_df_x[numerical_fea] = demo_df_x[numerical_fea].fillna(demo_df_x[numerical_fea].median())

demo_df_xx = demo_df_x[numerical_fea]

similarity_matrix = cosine_similarity(demo_df_xx)

sim_dict = {}
sim_train_dict = {}
for row_i in range(len(similarity_matrix)):
    xname = demos_data_list[row_i]
    if xname in demos_data_test_list:
#         continue
        row = similarity_matrix[row_i]
        sim_ids = np.argsort(-row)
        nears = []
        for sim_id in sim_ids:
            if row_i == sim_id:
                continue
            xx_name = demos_data_list[sim_id]
            if xx_name in demos_data_train_list:
                nears.append(xx_name)
        sim_dict[xname] = nears
    else:
        row = similarity_matrix[row_i]
        sim_ids = np.argsort(-row)
        nears = []
        for sim_id in sim_ids:
            if row_i == sim_id:
                continue
            xx_name = demos_data_list[sim_id]
            if xx_name in demos_data_train_list:
                nears.append(xx_name)
        sim_train_dict[xname] = nears
#     break
print (len(sim_dict))
print (len(sim_train_dict))

save_json(sim_dict, "output/final_sim_dict"+str(config['seed'])+".json")
save_json(sim_train_dict, "output/final_train_sim_dict"+str(config['seed'])+".json")

