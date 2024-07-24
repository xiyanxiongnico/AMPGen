import pandas as pd
df = pd.read_csv('/media/zzh/data/AMP/Transformerregression/data/5_65_stpa_mean_representations.csv')
num = len(df)
train_sample = df[:int(0.9 * num)]
# test_sample = df[int(0.6 * num):int(0.8 * num)]
last_sample = df[int(0.9 * num):]
print(train_sample.shape)
# print(test_sample.shape)
print(last_sample.shape)
train_sample.to_csv('/media/zzh/data/AMP/Transformerregression/data/train5_65_stpa_mean_representations.csv',index=False)
# test_sample.to_csv('/Users/zhangzheng/Desktop/xgboost_classifier/feature_generated/top14test_test.csv',index=False)
last_sample.to_csv('/media/zzh/data/AMP/Transformerregression/data/last5_65_stpa_mean_representations.csv',index=False)