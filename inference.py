import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier


def inference(in_data='./patient_1.csv', train_data='./healthcare-dataset-stroke-data.csv'):
    input_data = pd.read_csv(in_data)
    training_data = pd.read_csv(train_data)
    inference_data = pd.concat([input_data, training_data])
    inference_data = inference_data.drop(columns=['id'])
    inference_data = inference_data.apply(
        lambda column: LabelEncoder().fit_transform(column) if column.dtype == 'O' else column)
    inference_data = inference_data.drop(['stroke'], axis=1)
    inference_data = pd.DataFrame(StandardScaler().fit_transform(inference_data), columns=inference_data.columns)

    prediction_model = CatBoostClassifier()
    prediction_model.load_model("catboost")  # load our pre-trained model
    probability = prediction_model.predict_proba(inference_data)

    stroke = 0
    threshold = 0.95
    if probability[0][0] < threshold:
        stroke = 1

    return stroke


print(inference())
# the output is integer 0 or 1