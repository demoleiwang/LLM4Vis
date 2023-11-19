import pandas as pd
import random

def data_prepare():
    data_examples = pd.read_csv("dataset/example2_nodiscrization_3.csv")


    ## all used data examples
    demos_data = {
    'bar': 
    ['_Tyr:5', 'tompinder:13', 'futureboy:0', 'yoavweiss:0', 'ubun:12', 'alexisss:14', 'ayana.whiteheadsmith:2', 'rebeca.dioce:3', 'rjfleck:3', 'koushik.sbiiit:22', 'peluz:1', 'abhijitlondhe:22', 'ahmedsaab:12', 'airwickzoo:2', 'DavidJBianco:14', 'didiwai:247', 'sablanchard:2', 'thilina454d:37', 'babsndeep:0', 'VitasAnderson:1', 'gnedster:7', 'Jabawakai:45', 'jc041304:1', 'torikelly:1', 'knight00800:0', 'namkyu.park1:0', 'UladzislauVarabei:75', 'AlessiaT:4', 'devinp:7', 'hinkyeol.woo:10', 'matiya:3', 'rmirandarj:5', 'roadster11x:0', 'jevelezse:36', 'keithtxw:23', 'lucky11:4', 'FakePersona:2', 'hadyelsahar:0', 'javierramirez:4', 'cunyauthor:18'], 
    'line': 
    ['david24v:18', 'mckinstryt:4', 'matanatr96:9', 'RT03:1', '202438:3', 'athbyron:11', 'kmthakur:6', 'Maks:1', 'bloudermilk:7', 'dougkelly:31', 'f.attila16:5', 'gwinkler:0', 'jtbthomas2:0', 'lukesmolo92:0', 'maxbatt:5', 'max_tit:0', 'olemunch:476', 'scubacarlo:9', 'sdfgiuihjiuqaiu:2', 'sohrabrs:47', 'Frederikph:16', 'hadro:9', 'ines.chami:126', 'ats2660:3', 'janisz:2', 'Skee09:1', 'Chopper1901:186', 'robertomenezes:4', 'ashishkatnoria:5', 'sangee:169', 'paul.dylan.clark:2', 'Servinal:1', 'sidgoyal1:1', 'test12345:219', 'ylkjick532428:9', 'antrmendoza3:12', 'elovell1:3', 'IsaacCastromayor:3', 'GustavoMamaoAgnish:8', 'jchiuRP:4'], 
    'scatter': 
    ['alizauf:29', 'FatherMaldoon:6', 'jdu:52', 'kalpana74:79', 'KoenVerschuren:0', 'mrk-andreev:30', 'mszym:39', 'shawncaeiro:59', 'toegan:0', 'vedangw:6', 'weiyi:0', 'AlexDataScientist:38', 'dem0sh:35', 'espac1o:0', 'geet93:131', 'hari.sri.s:72', 'jbochi:11', 'justinm.hsi:2', 'kinson:6', 'maxyan:145', 'mingyangzhou:360', 'myan_berkeley:2', 'ramiroap1612:60', 'shijie.zhao.gt:298', 'Thanmai:17', 'tzs:157', 'a.elefsiniotis:0', 'aficionado:2', 'otkoge:0', 'acarlsson2:30', 'mehul.d:0', 'mzalaya:0', 'pat.vdleer:5', 'reuben.pereira126:48', 'snakedog:12', '204876:7', 'ChristopherEllis:12', 'Evan204826:15', 'jinglu921:1', 'sarvagyas12:6'], 
    'box': 
    ['jdoen1990:1', 'jhadley:36', 'sruthiravula:3', 'a.gudakov:89', 'nmming90:10', 'elizabeth1086:32', 'hongxu1013041980:30', 'alicia.b:5', 'HeatherVanGerven:1', 'Geraldo:13', 'ryan_v:11', 'dustyneggers:72', 'JonatanFigueroaGil:10', 'guitchounts:20', 'Laura.Fosso:21', 'avasquez:1', 'CassK:2', 'andreatm14:3', 'leesum:3', 'ahc72:2', 'bengin:42', '20.issareeya.b:9', 'ed.kk.ho:1', 'lyh:32', 'Runningman2:1', 'KcronoZz:2', 'kunal703:18', 'Lamanc:0', 'julia.hughes:5', 'mizanharris:51', '19naikittia.brayman:63', 'aymanmt:10', 'exatoa:9', 'zgcarvalho:5', 'nickdstew:1', 'xtralucky13:1', 'brianna.holland:7', 'pamela920:89', 'hzclinger:6', 'peecee:31']
    }

    demos_ids = {'line': [], 'scatter':[], 'bar':[], 'box':[]}
    for name in demos_data['line']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['line'].append(id_)
        
    for name in demos_data['scatter']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['scatter'].append(id_)

    for name in demos_data['bar']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['bar'].append(id_)

    for name in demos_data['box']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['box'].append(id_)


    demo_line_test_df = data_examples.iloc[demos_ids['line']]
    demo_scatter_test_df = data_examples.iloc[demos_ids['scatter']]
    demo_bar_test_df = data_examples.iloc[demos_ids['bar']]
    demo_box_test_df = data_examples.iloc[demos_ids['box']]

    demo_line_test_data_list = demo_line_test_df.to_dict("records")
    demo_scatter_test_data_list = demo_scatter_test_df.to_dict("records")
    demo_bar_test_data_list = demo_bar_test_df.to_dict("records")
    demo_box_test_data_list = demo_box_test_df.to_dict("records")

    demo_df = pd.concat([demo_line_test_df, demo_scatter_test_df, demo_bar_test_df, demo_box_test_df], ignore_index=True)
    final_data_list = demo_df.to_dict("records")


    selected_keys = list(demo_line_test_data_list[0].keys())[2:]

    target_dict = {'line':0, 'scatter':1, 'bar':2, 'box':3}

    return final_data_list, selected_keys, target_dict


def data_prepare_split(seed):

    data_examples = pd.read_csv("dataset/example2_nodiscrization_3.csv")

    demos_data = {'bar': ['_Tyr:5', 'tompinder:13', 'futureboy:0', 'yoavweiss:0', 'ubun:12', 'alexisss:14', 'ayana.whiteheadsmith:2', 'rebeca.dioce:3', 'rjfleck:3', 'koushik.sbiiit:22', 'peluz:1', 'abhijitlondhe:22', 'ahmedsaab:12', 'airwickzoo:2', 'DavidJBianco:14', 'didiwai:247', 'sablanchard:2', 'thilina454d:37', 'babsndeep:0', 'VitasAnderson:1', 'gnedster:7', 'Jabawakai:45', 'jc041304:1', 'torikelly:1', 'knight00800:0', 'namkyu.park1:0', 'UladzislauVarabei:75', 'AlessiaT:4', 'devinp:7', 'hinkyeol.woo:10', 'matiya:3', 'rmirandarj:5', 'roadster11x:0', 'jevelezse:36', 'keithtxw:23', 'lucky11:4', 'FakePersona:2', 'hadyelsahar:0', 'javierramirez:4', 'cunyauthor:18'], 'line': ['david24v:18', 'mckinstryt:4', 'matanatr96:9', 'RT03:1', '202438:3', 'athbyron:11', 'kmthakur:6', 'Maks:1', 'bloudermilk:7', 'dougkelly:31', 'f.attila16:5', 'gwinkler:0', 'jtbthomas2:0', 'lukesmolo92:0', 'maxbatt:5', 'max_tit:0', 'olemunch:476', 'scubacarlo:9', 'sdfgiuihjiuqaiu:2', 'sohrabrs:47', 'Frederikph:16', 'hadro:9', 'ines.chami:126', 'ats2660:3', 'janisz:2', 'Skee09:1', 'Chopper1901:186', 'robertomenezes:4', 'ashishkatnoria:5', 'sangee:169', 'paul.dylan.clark:2', 'Servinal:1', 'sidgoyal1:1', 'test12345:219', 'ylkjick532428:9', 'antrmendoza3:12', 'elovell1:3', 'IsaacCastromayor:3', 'GustavoMamaoAgnish:8', 'jchiuRP:4'], 'scatter': ['alizauf:29', 'FatherMaldoon:6', 'jdu:52', 'kalpana74:79', 'KoenVerschuren:0', 'mrk-andreev:30', 'mszym:39', 'shawncaeiro:59', 'toegan:0', 'vedangw:6', 'weiyi:0', 'AlexDataScientist:38', 'dem0sh:35', 'espac1o:0', 'geet93:131', 'hari.sri.s:72', 'jbochi:11', 'justinm.hsi:2', 'kinson:6', 'maxyan:145', 'mingyangzhou:360', 'myan_berkeley:2', 'ramiroap1612:60', 'shijie.zhao.gt:298', 'Thanmai:17', 'tzs:157', 'a.elefsiniotis:0', 'aficionado:2', 'otkoge:0', 'acarlsson2:30', 'mehul.d:0', 'mzalaya:0', 'pat.vdleer:5', 'reuben.pereira126:48', 'snakedog:12', '204876:7', 'ChristopherEllis:12', 'Evan204826:15', 'jinglu921:1', 'sarvagyas12:6'], 'box': ['jdoen1990:1', 'jhadley:36', 'sruthiravula:3', 'a.gudakov:89', 'nmming90:10', 'elizabeth1086:32', 'hongxu1013041980:30', 'alicia.b:5', 'HeatherVanGerven:1', 'Geraldo:13', 'ryan_v:11', 'dustyneggers:72', 'JonatanFigueroaGil:10', 'guitchounts:20', 'Laura.Fosso:21', 'avasquez:1', 'CassK:2', 'andreatm14:3', 'leesum:3', 'ahc72:2', 'bengin:42', '20.issareeya.b:9', 'ed.kk.ho:1', 'lyh:32', 'Runningman2:1', 'KcronoZz:2', 'kunal703:18', 'Lamanc:0', 'julia.hughes:5', 'mizanharris:51', '19naikittia.brayman:63', 'aymanmt:10', 'exatoa:9', 'zgcarvalho:5', 'nickdstew:1', 'xtralucky13:1', 'brianna.holland:7', 'pamela920:89', 'hzclinger:6', 'peecee:31']}

    # demos_data_train = {}
    # demos_data_test = {}
    # for k, v in demos_data.items():
    #     random.seed(seed)
    #     random.shuffle(v)
    #     if k not in demos_data_train:
    #         demos_data_train[k] = v[:15]
    #     if k not in demos_data_test:
    #         demos_data_test[k] = v[15:]
    demos_data_train = {'bar': ['javierramirez:4',
    'yoavweiss:0',
    'torikelly:1',
    'hadyelsahar:0',
    'jc041304:1',
    'koushik.sbiiit:22',
    'babsndeep:0',
    'ahmedsaab:12',
    'UladzislauVarabei:75',
    'thilina454d:37',
    'hinkyeol.woo:10',
    'sablanchard:2',
    'airwickzoo:2',
    'AlessiaT:4',
    'rjfleck:3'],
    'line': ['ashishkatnoria:5',
    'kmthakur:6',
    'elovell1:3',
    'david24v:18',
    'Skee09:1',
    'test12345:219',
    'sangee:169',
    'robertomenezes:4',
    'maxbatt:5',
    'IsaacCastromayor:3',
    'Maks:1',
    'Servinal:1',
    'sohrabrs:47',
    'hadro:9',
    'athbyron:11'],
    'scatter': ['toegan:0',
    'KoenVerschuren:0',
    'shawncaeiro:59',
    'mrk-andreev:30',
    'geet93:131',
    '204876:7',
    'Evan204826:15',
    'a.elefsiniotis:0',
    'sarvagyas12:6',
    'jinglu921:1',
    'maxyan:145',
    'mzalaya:0',
    'tzs:157',
    'ChristopherEllis:12',
    'FatherMaldoon:6'],
    'box': ['JonatanFigueroaGil:10',
    'hongxu1013041980:30',
    'pamela920:89',
    'Lamanc:0',
    'exatoa:9',
    'alicia.b:5',
    '19naikittia.brayman:63',
    'avasquez:1',
    'ahc72:2',
    'brianna.holland:7',
    'nmming90:10',
    'elizabeth1086:32',
    'Laura.Fosso:21',
    'hzclinger:6',
    'bengin:42']}

    demos_data_test = {'bar': ['peluz:1',
    'DavidJBianco:14',
    'didiwai:247',
    'lucky11:4',
    'namkyu.park1:0',
    'tompinder:13',
    'ubun:12',
    'rmirandarj:5',
    'futureboy:0',
    'abhijitlondhe:22',
    'keithtxw:23',
    '_Tyr:5',
    'FakePersona:2',
    'matiya:3',
    'jevelezse:36',
    'alexisss:14',
    'roadster11x:0',
    'VitasAnderson:1',
    'rebeca.dioce:3',
    'ayana.whiteheadsmith:2',
    'Jabawakai:45',
    'gnedster:7',
    'cunyauthor:18',
    'devinp:7',
    'knight00800:0'],
    'line': ['GustavoMamaoAgnish:8',
    'f.attila16:5',
    'jtbthomas2:0',
    'olemunch:476',
    'matanatr96:9',
    'dougkelly:31',
    'jchiuRP:4',
    'max_tit:0',
    'ylkjick532428:9',
    'mckinstryt:4',
    'ats2660:3',
    'sidgoyal1:1',
    'RT03:1',
    'ines.chami:126',
    'Frederikph:16',
    'Chopper1901:186',
    'scubacarlo:9',
    'sdfgiuihjiuqaiu:2',
    'antrmendoza3:12',
    'gwinkler:0',
    'lukesmolo92:0',
    'paul.dylan.clark:2',
    'janisz:2',
    'bloudermilk:7',
    '202438:3'],
    'scatter': ['reuben.pereira126:48',
    'justinm.hsi:2',
    'pat.vdleer:5',
    'acarlsson2:30',
    'kalpana74:79',
    'myan_berkeley:2',
    'snakedog:12',
    'mingyangzhou:360',
    'otkoge:0',
    'jdu:52',
    'mehul.d:0',
    'ramiroap1612:60',
    'hari.sri.s:72',
    'shijie.zhao.gt:298',
    'Thanmai:17',
    'espac1o:0',
    'mszym:39',
    'aficionado:2',
    'jbochi:11',
    'weiyi:0',
    'dem0sh:35',
    'vedangw:6',
    'kinson:6',
    'alizauf:29',
    'AlexDataScientist:38'],
    'box': ['jhadley:36',
    'Geraldo:13',
    'aymanmt:10',
    'kunal703:18',
    'CassK:2',
    'julia.hughes:5',
    'peecee:31',
    'nickdstew:1',
    'zgcarvalho:5',
    '20.issareeya.b:9',
    'HeatherVanGerven:1',
    'mizanharris:51',
    'ryan_v:11',
    'xtralucky13:1',
    'ed.kk.ho:1',
    'jdoen1990:1',
    'KcronoZz:2',
    'lyh:32',
    'Runningman2:1',
    'a.gudakov:89',
    'leesum:3',
    'sruthiravula:3',
    'guitchounts:20',
    'andreatm14:3',
    'dustyneggers:72']}

    ###
    demos_ids = {'line': [], 'scatter':[], 'bar':[], 'box':[]}
    for name in demos_data_train['line']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['line'].append(id_)
        
    for name in demos_data_train['scatter']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['scatter'].append(id_)

    for name in demos_data_train['bar']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['bar'].append(id_)

    for name in demos_data_train['box']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        demos_ids['box'].append(id_)

    ###
    correct_test_ids = {'line': [], 'scatter':[], 'bar':[], 'box':[]}
    for name in demos_data_test['line']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        correct_test_ids['line'].append(id_)

    for name in demos_data_test['scatter']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        correct_test_ids['scatter'].append(id_)

    for name in demos_data_test['bar']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        correct_test_ids['bar'].append(id_)

    for name in demos_data_test['box']:
        df_ = data_examples.loc[data_examples['fid'] == name]
        id_ = df_.index[0]
        correct_test_ids['box'].append(id_)

    ###
    demo_line_df = data_examples.iloc[demos_ids['line']]
    demo_scatter_df = data_examples.iloc[demos_ids['scatter']]
    demo_bar_df = data_examples.iloc[demos_ids['bar']]
    demo_box_df = data_examples.iloc[demos_ids['box']]

    demo_line_data_list = demo_line_df.to_dict("records")
    demo_scatter_data_list = demo_scatter_df.to_dict("records")
    demo_bar_data_list = demo_bar_df.to_dict("records")
    demo_box_data_list = demo_box_df.to_dict("records")


    ###
    line_test_df = data_examples.iloc[correct_test_ids['line']]
    scatter_test_df = data_examples.iloc[correct_test_ids['scatter']]
    bar_test_df = data_examples.iloc[correct_test_ids['bar']]
    box_test_df = data_examples.iloc[correct_test_ids['box']]

    line_test_data_list = line_test_df.to_dict("records")
    scatter_test_data_list = scatter_test_df.to_dict("records")
    bar_test_data_list = bar_test_df.to_dict("records")
    box_test_data_list = box_test_df.to_dict("records")

    selected_keys = list(line_test_data_list[0].keys())[4:]

    demo_data_list = demo_line_data_list + demo_scatter_data_list + demo_bar_data_list + demo_box_data_list
    test_data_list = line_test_data_list + scatter_test_data_list + bar_test_data_list + box_test_data_list

    return demo_data_list, test_data_list, selected_keys




