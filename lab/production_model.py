import pandas as pd
import joblib


# import model and samples
model_rf = joblib.load('lab/model_rf.pkl')
X_df_100 = pd.read_csv('lab/data/cardio_test_samples.csv', index_col='id')

# creating new dataframe to collect results, keeping 'id' as index
df_result = pd.DataFrame(index = X_df_100.index)

# populating 3 new columns with results from predictioin
df_result[['proba_class_0', 'proba_class_1']] = model_rf.predict_proba(X_df_100)
df_result['prediction'] = model_rf.predict(X_df_100)

df_result.to_csv('lab/data/cardio_result.csv')