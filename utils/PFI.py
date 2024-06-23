import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def PFI(X, labels, model, base_rmse: float):
  results = []
  f_names = []

  for feature in X:
    f_names.append(feature)
    # Create a copy of X_test
    X_copy = X.copy(deep = True).reset_index(drop=True)

    # Scramble the values of the given predictor
    X_copy[feature] = X_copy[feature].sample(frac=1).reset_index(drop=True)
    # Calculate the new RMSE
    new_preds = model.predict(X_copy)

    new_RMSE = mean_squared_error(labels, new_preds)

    # Append the increase in MSE to the list of results
    results.append(new_RMSE - base_rmse)
  print(results)

  # Put the results into a pandas dataframe and rank the predictors by score
  results_df = pd.DataFrame(
      {
          'feature' : f_names,
          'RMSEChange' : results
      }

  ).sort_values(by = 'RMSEChange', ascending = False)
  return results_df