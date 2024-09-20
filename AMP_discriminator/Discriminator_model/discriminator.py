
"""
train the classifier:
train_amp_classifier --data_path path/to/classify_all_data_v1.csv --model_output_path path/to/save/xgboost_model.pkl

Command-Line Usage
classify_amp --train_path path/to/classify_all_data_v1.csv --pre_path path/to/new_sequences.csv --out_path path/to/save/predictions.csv
"""

import xgboost
from sklearn import metrics
import pandas as pd
from features import get_pre_features
from sklearn.model_selection import train_test_split
import argparse

def classify_sequences(train_path, pre_path, out_path, to_device):
    """
    Classify sequences using a pre-trained XGBoost model and save the results to a CSV file.
    :param train_path: Path to the training CSV file
    :param pre_path: Path to the input CSV file containing sequences and their properties
    :param out_path: Path to the output CSV file for saving classification results
    """
    data = pd.read_csv(train_path)
    X = data.iloc[:, 2:-1].values
    Y = data.iloc[:, -1].values
    x_train0, x_test, y_train0, y_test = train_test_split(X, Y, test_size=0.3, random_state=77)
    
    lr = 0.1
    md = 5
    ne = 2000
    # Define the model
    model = xgboost.XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr, objective="binary:logistic", tree_method="hist", use_label_encoder=False, eval_metric="logloss", device=to_device)
    model.fit(x_train0, y_train0)

    # Load and preprocess candidate data
    candidate_data = get_pre_features(pre_path)
    x_data = candidate_data.iloc[:, 1:].values
    pred = model.predict(x_data)
    pred = pd.DataFrame(pred, columns=['type'])
    itms = pd.read_csv(pre_path)
    result = pd.concat([itms, pred], axis=1)

    # Save the results
    result.to_csv(out_path, index=False)
    print(f"Classification results saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify sequences using a pre-trained XGBoost model")
    parser.add_argument('--train_path', '-tp', required=True, help='Path to the training CSV file')
    parser.add_argument('--pre_path', '-pp', required=True, help='Path to the input CSV file containing sequences and their properties')
    parser.add_argument('--out_path', '-op', required=True, help='Path to the output CSV file for saving classification results')
    parser.add_argument('--to_device', type=str, default="cuda", help="Device to run the model, cuda or cpu")

    args = parser.parse_args()
    classify_sequences(args.train_path, args.pre_path, args.out_path, args.to_device)
    
    
    
    
    
    
    
    