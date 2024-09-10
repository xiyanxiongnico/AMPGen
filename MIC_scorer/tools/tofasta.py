import pandas as pd 
df = pd.read_csv('/Users/zhangzheng/Desktop/AMP_temp/train_data/regression/ave_5-65regression_stpa_all.csv')
seq_list = df['Sequence']
seq_list = seq_list.tolist()

number = len(seq_list) + 1
name = [i for i in range(1,number)]
# Define a function to format sequences in FASTA format

with open('/Users/zhangzheng/Desktop/test/5_65_stpa.fasta', 'w') as fasta_file:
    for i in name:
        fasta_lines = '>'+str(i)
        j = i - 1
        seq_lines = seq_list[j]
        # print(fasta_lines)
        fasta_file.write(fasta_lines)
        fasta_file.write('\n')
        fasta_file.write(seq_lines)
        fasta_file.write('\n')
