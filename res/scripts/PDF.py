import numpy as np

def PDF(X, model, feature):
  """
  Input   Dataset 'X', Model 'model', feature_name 'feature'
  Output  x_values: independent variable values
          f_values: corresponding output per x value
  """
  # Note: uncomment the lines below and complete the right hand side (where you see '..' to set them to suitable values, respective explanations are provided for each variable

  X = X.copy(deep = True) # prevents modifying the original dataframe
  assert feature in list(X.columns) # sanity check

  num_samples = 100  # set the number of samples/steps to slice the range of the continuous feature, e.g., 100.
  min_val = X[feature].min()     # minimum value of the given feature
  max_val = X[feature].max()      # maximum value of the given feature
  step_size =  (max_val-min_val)/(num_samples)   # see the algorithm in corresponsing lecture slides to calculate the step size as a function of the above variables

  x_values = np.arange(start = min_val,
                       stop = max_val,
                       step = step_size)     # x_values at which we will calculate the partial function of the given feature
  f_values = np.empty(shape = x_values.shape)     # the calculated partial function values corresponding to x_values



  for k in range(num_samples - 1):
    # Change part of the data according to the formula of PDF algorithm
    # Let the model predict and calculate the f_value for this k
    X[feature] = np.full(X[feature].shape, x_values[k])
    f_values[k] = np.average(model.predict(X))



  return x_values, f_values