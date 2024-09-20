

from iFeature.codes import *
from iFeature.PseKRAAC import *
import numpy as np
import pandas as pd
from tqdm import tqdm

def isDataFrame(obj):
    if type(obj) == pd.core.frame.DataFrame:
        return True
    else:
        return False

def getTrueFastas(fastas):
    if isDataFrame(fastas):
        fts = fastas[["ID", "Sequence"]]
        return fts.to_numpy()
    else:
        return fastas

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
def one_hot_encode(seq, max_len=60):
    # o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    l = len(s)
    if max_len < l:
        max_len = l
    x = pd.DataFrame(np.zeros((max_len, 20), dtype=int), columns=codes)
    a = s[0].str.get_dummies(sep=',')
    # a = a.join(x)
    # a = a.sort_index(axis=1)
    b = x + a
    b = b.replace(np.nan, 0)
    b = b.astype(dtype=int)
    # e = a.values.flatten()
    return b

def oneHot(fastas, max_len=60, class_val=None):
    fastas = getTrueFastas(fastas)
    fts = []
    names = fastas[:,0]
    for seq in fastas[:,1]:
        if len(seq) > max_len:
            continue
        # print("seq: ", seq)
        e = one_hot_encode(seq, max_len=max_len)
        e = e.values.flatten()
        fts.append(e)
    df = pd.DataFrame(fts)
    df.index = names
    # df.columns = df.iloc[0]
    print("Gene oneHot:")
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

#subtype = {'g-gap': 0, 'lambda-correlation': 4}
#subtype = 'g-gap' or 'lambda-correlation'
def genePsekraac(fastas, ft_name="type1", raactype=2, subtype='lambda-correlation', ktuple=2, gap_lambda=1, class_val=None):
    """
    run codes as follows if you want try all kraac features of ifeature.PseKRAAC
    from codes.ifeature.PseKRAAC import *
    from codes.ifeature.codes import readFasta
    for ft_name in kraac:
        AAGroup = eval("%s.AAGroup" % ft_name)
        for raactype in AAGroup:
    :param path: given fasta file location
    :param ft_name: one element of the following list:
        ["type1","type2","type3A","type3B","type4","type5","type6A",
         "type6B","type6C","type7","type8","type9","type10","type11",
         "type12","type13","type14","type15","type16"]
    :param raactype: key of AAGroup # of ifeature.PseKRAAC.typeX
    :param subtype: a dictionary, for example: {'g-gap': 0, 'lambda-correlation': 4}
        Note: g-gap + lambda-correlation < 5 (smallest length of sequences of given fasta file)
    :param ktuple: int 1, 2, or 3 # of ifeature.PseKRAAC.typeX
    :param gap_lambda: int 1 # of ifeature.PseKRAAC.typeX
        gap value for the ‘g-gap’ model  or lambda value for the ‘lambda-correlation’ model, 10 values are available (i.e. 0, 1, 2, ..., 9)
    :return: dataframe of feature and the last column is the label of class with name "default"
    """
    # fastas = readFasta.readFasta(path)
    # gap_lambda = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    fastas = getTrueFastas(fastas)
    #type1(fastas, subtype, raactype, ktuple, glValue)
    eval_func = "%s.type1(fastas, subtype=subtype, raactype=raactype, ktuple=ktuple, glValue=gap_lambda)" % (ft_name)
    # print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    # print("feature number of PseKRAAC.%s(%s, raac_type=%d, ktuple=%d, gap_lambda=%d): %d" %
    #       (ft_name, subtype, raactype, ktuple, gap_lambda, len(df.columns)))
    ft_whole_name = "%sraac%s" % (ft_name, raactype)
    # print("Gene %s :" % (ft_whole_name))
    row, col = df.shape
    # print("\t\t its class value: %s" % (str(class_val)))
    # print("\t\t its input No.: %s" % row)
    # print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def GeneIfeature(fastas, ft_name="AAC", gap=0, nlag=4, lambdaValue=4, class_val=None):
    """
    run codes as follows if you want try all features of ifeature.codes:
    from codes.ifeature.codes import *
    for ft_name in ft_whole_type:
        if ft_name in rm_type:
            continue
    :param path: given fasta file location
    :param ft_name: each element of a list ft_whole_type:
            ft_whole_type = ["AAC","EAAC","CKSAAP","DPC","DDE","TPC","BINARY","GAAC","EGAAC",
                "CKSAAGP","GDPC","GTPC","AAINDEX","ZSCALE","BLOSUM62","NMBroto",
                "Moran","Geary","CTDC","CTDT","CTDD","CTriad","KSCTriad",
                "SOCNumber","QSOrder","PAAC","APAAC","KNNprotein","KNNpeptide",
                "PSSM","SSEC","SSEB","Disorder","DisorderC","DisorderB","ASA","TA"]
            except the list rm_type as follows:
            rm_type = ["EAAC", "BINARY", "EGAAC", "AAINDEX", "ZSCALE", "BLOSUM62", "PSSM", "ASA", "TA", "Disorder", "DisorderB",
                   "DisorderC", "KNNprotein", "KNNpeptide", "SSEC", "SSEB"]
            because of errors when run ifeature.codes.ft_name:
                Error: for "EAAC"/"BINARY"/"EGAAC"/"AAINDEX"/"ZSCALE"/"BLOSUM62"/"PSSM"/"ASA"/"TA" encoding, the input fasta sequences should be with equal length.
                "KNNprotein"/"KNNpeptide": should have the train fasta file and a label file, do it later
                "SSEC"/"SSEB": secondary structure
                "Disorder"/"DisorderB"/"DisorderC" : Protein disorder information was first predicted by the VSL2 software
                "Disorder"/"DisorderB": encoding, the input fasta sequences should be with equal length.
    :return: dataframe of feature and the last column is the label of class with name "default"
    """
    # fastas = readFasta.readFasta(path)
    # CKSAAP: gap = 0, 1, 2, 3 (3 = min sequence length - 2)
    # SOCNumber QSOrder PAAC APAAC: lambdaValue = 0, 1, 2, 3, 4 (4 = min sequence length - 1)
    # NMBroto: nlag= 2, 3, 4
    fastas = getTrueFastas(fastas)
    eval_func = "%s.%s(fastas, gap=%d, order=None, nlag=%d, lambdaValue=%d)" % (ft_name, ft_name, gap, nlag, lambdaValue)
    # print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    # print("%s's feature number: %d" % (ft_name, len(df.columns)))
    # print("Gene %s :" % (ft_name))
    row, col = df.shape
    # print("\t\t its class value: %s" % (str(class_val)))
    # print("\t\t its input No.: %s" % row)
    # print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def GeneftFromSequences(fastas, ft_whole_name="type8raac18", nlag=4, lambdaValue=4,
                        subtype={'g-gap': 0, 'lambda-correlation': 4},
                         ktuple=2, gap_lambda=1, class_val=None):
    # s = GetSpecieInfo(specie_name)
    if "type" in ft_whole_name:
        raactype = int(ft_whole_name.split("raac")[-1])
        if "Ktuple" in ft_whole_name:
            ft_name = ft_whole_name.split("Ktuple")[0]
        else:
            ft_name = ft_whole_name.split("raac")[0]
        ft = genePsekraac(fastas, ft_name=ft_name, raactype=raactype, subtype=subtype, ktuple=ktuple, gap_lambda=gap_lambda, class_val=class_val)
    else:
        ft_name = ft_whole_name
        ft = GeneIfeature(fastas, ft_name=ft_name, nlag=nlag, lambdaValue=lambdaValue, class_val=class_val)
    return ft

# low, high means raac type range [low, high]
def geneAllFeatureNames(low=1, high=19):
    raac_name = []
    ft_names = []
    kraacs = ["type1", "type2", "type3A", "type3B", "type4", "type5", "type6A",
              "type6B", "type6C", "type7", "type8", "type9", "type10", "type11",
              "type12", "type13", "type14", "type15", "type16"]
    # PseKRAAC: gap_lambda = list(range(10))
    ift_types = ["AAC", "DPC", "DDE", "TPC", "GAAC", "GDPC", "GTPC", "CTDC", "CTDT", "CTDD"]
    # CKSAAP: gap = 0, 1, 2, 3 (3 = min sequence length - 2)
    # SOCNumber QSOrder PAAC APAAC: lambdaValue = 0, 1, 2, 3, 4 (4 = min sequence length - 1)
    # NMBroto: nlag= 2, 3, 4
    ft_types = []
    for ft in ["CKSAAP", "CKSAAGP", "CTriad", "KSCTriad"]:
        for gap in range(4):
            if ft == "KSCTriad" and gap > 1:
                continue
            ft_types.append("%s_gap%d" % (ft, gap))
    for ft in ["SOCNumber", "QSOrder", "PAAC", "APAAC"]:
        for lambdaValue in range(5):
            ft_types.append("%s_lmd%d" % (ft, lambdaValue))
    for ft in ["Moran", "NMBroto", "Geary"]:
        for nlag in range(2,5,1):
            ft_types.append("%s_nlag%d" % (ft, nlag))
    for ft_type in kraacs:
        AAGroup = eval("%s.AAGroup" % ft_type)
        for raactype in AAGroup:
            if raactype in list(range(low, high+1)):
                for gap_lambda in range(10):
                    for subtype in ["g-gap", "lambda-correlation"]:
                        if subtype == "lambda-correlation" and gap_lambda == 0:
                            continue
                        raac_name.append("%sraac%dglmd%d%s" % (ft_type, raactype, gap_lambda, subtype))
    ft_names.extend(ift_types)
    ft_names.extend(ft_types)
    ft_names.append("oneHot")
    ft_names.extend(raac_name)
    return ft_names

def geneAllFeatures(fastas, low=1, high=19, max_len=60):
    fts_info = []
    # gene OneHot:
    fts = oneHot(fastas, max_len=max_len)
    ft_whole_name = "oneHot"
    colname = ["name", "number"]
    ft_info = [ft_whole_name, fts.shape[1]]
    fts_info.append(ft_info)
    # fts = GeneIfeature(fastas, ft_name="AAC", gap=0, nlag=4, lambdaValue=4, class_val=1)
    # Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3).
    # Gene ifts:
    ift_types = ["AAC", "DPC", "DDE", "TPC", "GAAC", "GDPC", "GTPC", "CTDC", "CTDT", "CTDD"]
    for ft_name in ift_types:
        fts = GeneIfeature(fastas, ft_name=ft_name)
        fts_info.append([ft_name, fts.shape[1]])
    ift_types_gap = ["CKSAAP", "CKSAAGP", "CTriad", "KSCTriad"]
    for ft_name in ift_types_gap:
        for gap in range(4):
            if ft_name == "KSCTriad" and gap > 1:
                continue
            ft_whole_name = "%s_gap%d" % (ft_name, gap)
            fts = GeneIfeature(fastas, ft_name=ft_name, gap=gap)
            fts_info.append([ft_whole_name, fts.shape[1]])
    for ft in ["SOCNumber", "QSOrder", "PAAC", "APAAC"]:
        for lambdaValue in range(5):
            ft_whole_name = "%s_lmd%d" % (ft, lambdaValue)
            fts = GeneIfeature(fastas, ft_name=ft_name, lambdaValue=lambdaValue)
            fts_info.append([ft_whole_name, fts.shape[1]])
    for ft in ["Moran", "NMBroto", "Geary"]:
        for nlag in range(2,5,1):
            ft_whole_name = "%s_nlag%d" % (ft, nlag)
            fts = GeneIfeature(fastas, ft_name=ft_name, nlag=nlag)
            fts_info.append([ft_whole_name, fts.shape[1]])
    kraacs = ["type1", "type2", "type3A", "type3B", "type4", "type5", "type6A",
              "type6B", "type6C", "type7", "type8", "type9", "type10", "type11",
              "type12", "type13", "type14", "type15", "type16"]
    for ft_type in kraacs:
        AAGroup = eval("%s.AAGroup" % ft_type)
        for raactype in AAGroup:
            if raactype in list(range(low, high+1)):
                for gap_lambda in range(10):
                    for subtype in ["g-gap", "lambda-correlation"]:
                        if subtype == "lambda-correlation" and gap_lambda == 0:
                            continue
                        ft_whole_name = "%sraac%dglmd%d%s" % (ft_type, raactype, gap_lambda, subtype)
                        fts = genePsekraac(fastas, ft_name=ft_type, raactype=raactype, subtype=subtype,
                                     ktuple=2, gap_lambda=gap_lambda)
                        fts_info.append([ft_whole_name, fts.shape[1]])
    df = pd.DataFrame(fts_info, columns=colname)
    df.to_csv("features_info.txt")
    print("print features used in the features_info.txt")
    print("total %d features used in this code." % len(df))
    fts_info = pd.DataFrame(fts_info)
    fts_info.to_csv("./features_number.txt", index=False)
    return fts_info

def geneFeature(fastas, ft_whole_name, max_len=60):
    # fts = GeneIfeature(fastas, ft_name="AAC", gap=0, nlag=4, lambdaValue=4, class_val=1)
    # Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3).
    # Gene ifts:
    ift_types = ["AAC", "DPC", "DDE", "TPC", "GAAC", "GDPC", "GTPC", "CTDC", "CTDT", "CTDD"]
    ift_types_gap = ["CKSAAP", "CKSAAGP", "CTriad", "KSCTriad"]
    if "type" in ft_whole_name:
        ft_list = ft_whole_name.split("raac")
        ft_type = ft_list[0]
        ft_list = ft_list[1].split("glmd")
        raactype = int(ft_list[0])
        if "g-gap" in ft_list[1]:
            gap_lambda = int(ft_list[1].split("g-gap")[0])
            subtype = "g-gap"
        else:
            gap_lambda = int(ft_list[1].split("lambda-correlation")[0])
            subtype = "lambda-correlation"
        ft_para = {"ft_type": ft_type, "raactype": raactype, "subtype": subtype, "gap_lambda": gap_lambda, "psekraac":True}
        # print("processing fastas: \n", fastas)
        fts = genePsekraac(fastas, ft_name=ft_type, raactype=raactype, subtype=subtype,
                           ktuple=2, gap_lambda=gap_lambda)
    elif "_" in ft_whole_name:
        ft_list = ft_whole_name.split("_")
        ft_type = ft_list[0]
        para = ft_list[1]
        if "gap" in para:
            gap = int(para.split("gap")[1])
            ft_para = {"ft_type": ft_type, "gap": gap, "psekraac": False}
            fts = GeneIfeature(fastas, ft_name=ft_type, gap=gap)
        elif "lmd" in para:
            lambdaValue = int(para.split("lmd")[1])
            ft_para = {"ft_type": ft_type, "lambdaValue": lambdaValue, "psekraac": False}
            fts = GeneIfeature(fastas, ft_name=ft_type, lambdaValue=lambdaValue)
        elif "nlag" in para:
            nlag = int(para.split("nlag")[1])
            ft_para = {"ft_type": ft_type, "nlag": nlag, "psekraac": False}
            fts = GeneIfeature(fastas, ft_name=ft_type, nlag=nlag)
    elif ft_whole_name == "oneHot":
        fts = oneHot(fastas, max_len=max_len)
        ft_para = {"ft_type": "oneHot"}
    elif ft_whole_name in ift_types:
        fts = GeneIfeature(fastas, ft_name=ft_whole_name)
        ft_para = {"ft_type": ft_whole_name}
    # print(ft_whole_name)
    # print("features and its parameters used: \n", ft_para)
    # print("features size: \n\trow: %d \n\tcol: %d" % fts.shape)
    return fts

def geneFeatures(fastas, ft_names, max_len=60):
    for i, ft_name in enumerate(ft_names):
        ft = geneFeature(fastas=fastas, ft_whole_name=ft_name, max_len=max_len)
        row = ft.shape[0]
        if ft_name == "AAC":
            ft = np.hstack([ft] * 20).reshape(row, -1)
        # if ft_name == "DPC":
        #     ft = np.hstack([ft] * 20).reshape(row, -1)
        if i == 0:
            fts = ft
        else:
            fts = np.hstack((fts, ft))
    return fts

def get_train_features(path):
    test_df = pd.read_csv(path)
    seq_df = test_df.iloc[:,1:2]
    type_df = test_df.iloc[:,-1] ### for generated sequence type is not sure 
    whole_name =  \
    ['type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2',
    'QSOrder_lmd1', 'QSOrder_lmd0', 'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
    'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation',
    'type2raac8glmd2lambda-correlation', 'type8raac14glmd1lambda-correlation']
    features_list = []
    for ft_whole_name in tqdm(whole_name):    
        test_feature_df = geneFeature(test_df,ft_whole_name)
        test_feature_df = test_feature_df.iloc[:,1:]
        features_list.append(test_feature_df)
    df = pd.concat(features_list,axis=1)
    df_all = pd.concat([seq_df, df, type_df],axis=1)
    df_all.to_csv('top14test.csv')
    return df_all

def get_pre_features(path):
    test_df = pd.read_csv(path)
    seq_df = test_df.iloc[:,1:2]
    whole_name =  \
    ['type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2',
    'QSOrder_lmd1', 'QSOrder_lmd0', 'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
    'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation',
    'type2raac8glmd2lambda-correlation', 'type8raac14glmd1lambda-correlation']
    features_list = []
    for ft_whole_name in tqdm(whole_name):    
        test_feature_df = geneFeature(test_df,ft_whole_name)
        test_feature_df = test_feature_df.iloc[:,1:]
        features_list.append(test_feature_df)
    df = pd.concat(features_list, axis=1)
    df_all = pd.concat([seq_df, df], axis=1)
    return df_all

if __name__ == '__main__':
    path = '/media/zzh/data/AMP/xgboost_classifier/data/test.csv'
    res = get_train_features(path)
    df_all.to_csv('/media/zzh/data/AMP/xgboost_classifier/data/top14test.csv')
    
    
    

    
