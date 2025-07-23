"""
This program can read and manipulate NYC school data,
create a linear models to predict a school's graduation rates,
and evaluate that model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(file_name):
    """Read relevant data from DOE High School Directory CSV file.
    This function should only select the following columns, preserving their order:
    [
        "dbn",
        "school_name",
        "borocode",
        "NTA",
        "graduation_rate",
        "pct_stu_safe",
        "attendance_rate",
        "college_career_rate",
        "language_classes",
        "advancedplacement_courses"
    ]
    Any rows with missing values for `graduation rate`
    should be dropped from the returned dataframe.
    
    :param file_name: name/path to DOE High School Directory CSV file.
    :type file_name: str.
    :returns: relevant data from specified CSV file.
    :rtype: pandas.DataFrame.
    """
    columns = [ "dbn", "school_name", "borocode",
                "NTA", "graduation_rate", "pct_stu_safe",
                "attendance_rate","college_career_rate",
                "language_classes", "advancedplacement_courses"
                ]
    hs_df = pd.read_csv(file_name, usecols=columns)[columns] # to preserve order
    hs_df = hs_df.dropna(subset=['graduation_rate'])
    return hs_df


def impute_numeric_cols_median(df):
    """Impute missing numeric values with column median.
     Any missing entries in the numeric columns
     ['pct_stu_safe','attendance_rate', 'college_career_rate']
     are replaced with the median of the respective column's non-missing values.

    :param df: dataframe ontaining DOE High School from OpenData NYC.
    :type df: pandas.DataFrame.
    :returns: dataframe with imputed values.
    :rtype: pandas.DataFrame.
    """
    medians = {"pct_stu_safe": df['pct_stu_safe'].median(),
               "attendance_rate": df['attendance_rate'].median(),
               "college_career_rate": df['college_career_rate'].median()}
    return df.fillna(value=medians)

def compute_item_count(df, d_col):
    """Counts the number of items, separated by commas, in each entry of `df[col]`.
    
    :param df: dataframe containing DOE High School from OpenData NYC.
    :type df: pandas.DataFrame.
    :col: a column key in `df` that contains a list of items separated by 
        commas. 
    :type d_col: str.
    :returns series with
    :rtype: pandas.Series.
    """
    count_list = []
    for items in df[d_col]:
        if pd.isna(items):
            count_list.append(0)
        else:
            count_list.append(items.count(',') + 1)
    return pd.Series(data=count_list, name=d_col)

def encode_categorical_col(x):
    """One-hot encode a categorical column.
    
    Takes a column of categorical data and performs one-hot encoding to create a 
    new DataFrame with k columns, where k is the number of distinct values in the 
    column. Output columns should have the same ordering as their values would if
    sorted.
    
    :param x: series containing categorical data.
    :type x: pandas.Series.
    :returns: dataframe of categorical encodings of x.
    :rtype: pandas.DataFrame.
    """
    # one-hot encoding: 0 if absent, 1 if present for every
    unique_columns = []
    for x_val in x:
        for x_item in x_val.split(', '):
            if x_item not in unique_columns:
                unique_columns.append(x_item)
    unique_columns.sort()
    # index = [x for x in range(1,len(x)+1)]
    encoded_df = pd.DataFrame(columns=unique_columns, index = x.index)
    for uniq in unique_columns:
        encoded_list = []
        for x_val in x:
            if uniq in x_val:
                encoded_list.append(1)
            else:
                encoded_list.append(0)
        encoded_df[uniq] = encoded_list
    return encoded_df

def split_test_train(df, x_col_names, y_col_name, frac=0.25, random_state=922):
    """Split data into train and test subsets.
    :param df: dataframe containing input columns (aka independent variables, 
        predictors, features, covariates, ...) and output column (aka dependent 
        variable, target, ...).
    :type df: pandas.DataFrame.
    :param x_col_names: column keys to input variables.
    :type x_col_names: list or iterable.
    :param y_col_name: column key to output variable.
    :type y_col_name: str.
    :param frac: fraction (between 0 and 1) of the data for the test set. Defaults to 0.25.
    :type frac: float.
    :param random_state: random generator seed. Defaults to 922.
    :type random_state: int.
    :returns: a tuple (x_train, x_test, y_train, y_test) with selected columns
    of the original data in df split into train and test sets.
    :rtype: tuple(pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series)
    """
    test_df = df.sample(frac=frac, random_state=random_state)
    # print(test_df)
    train_df = df.copy()
    drop_indices = test_df.index.to_list()
    # print("drop_indices: " + str(drop_indices))
    train_df = train_df.drop(axis=0, index=drop_indices)
    return train_df[x_col_names], test_df[x_col_names], train_df[y_col_name], test_df[y_col_name]


# Section 1: Data ingestion and feature engineering

FILE_NAME = '2021_DOE_High_School_Directory_SI.csv'
si_df = import_data(FILE_NAME)
print(f'There are {len(si_df.columns)} columns:')
print(si_df.columns)
print('The dataframe is:')
print(si_df)

FILE_NAME = '2020_DOE_High_School_Directory_late_start.csv'
late_df = import_data(FILE_NAME)
print('The numerical columns are:')
print(late_df[['dbn','pct_stu_safe','attendance_rate','college_career_rate']])

late_df = impute_numeric_cols_median(late_df)
print(late_df[['dbn','pct_stu_safe','attendance_rate','college_career_rate']])

# Now, using the `compute_item_count` twice,
# add two new columns with counts for languages \& AP classes:
late_df['language_count'] = compute_item_count(late_df,'language_classes')
late_df['ap_count'] = compute_item_count(late_df,'advancedplacement_courses')
print('High schools that have 9am or later start:')
print(late_df[['dbn','language_count','language_classes','ap_count','advancedplacement_courses']])

# Now add columns for the borough code, using one hot encoding:
boros_df = encode_categorical_col(late_df['borocode'])
print(late_df['borocode'].head(5))
print(boros_df.head(5))

# gives a new DataFrame with `0` and `1` values.
# Check your work by counting number of schools in each borough:
print('Number of schools in each borough:')
print(boros_df.sum(axis=0))

x_cols = ['language_count','ap_count','pct_stu_safe','attendance_rate','college_career_rate']
Y_COL = 'graduation_rate'
x_train, x_test, y_train, y_test = split_test_train(late_df, x_cols, Y_COL)
print('The sizes of the sets are:')
print(f'x_train has {len(x_train)} rows.\tx_test has {len(x_test)} rows.')
print(f'y_train has {len(y_train)} rows.\ty_test has {len(y_test)} rows.')

# Section 2: Training a linear regressor
def compute_lin_reg(x, y):
    """
    :param x: 1-dimensional array containing the predictor (independent) variable values.
    :type x: pandas.Series, numpy.ndarray, or iterable of numeric values.
    :param y: 1-dimensional array containing the target (dependent) variable values.
    :type y: pandas.Series, numpy.ndarray, or iterable of numeric values.
    :return: tuple containing the model's (intercept, slope)
    :rtype: tuple(float, float)

     The function computes the slope and y-intercept of the 1-d linear regression line, 
     using ordinary least squares (see DS 8, Chapter 15 or DS 100, Chapter 15 for detailed 
     explanation.
     
     Algorithm for this:
        1. Compute the standard deviation of `x` and `y`. Call these `sd_x` and `sd_y`.
        2. Compute the correlation, `r`, between `x` and `y`.
        3. Compute the slope, `theta_1`, as `theta_1 = r*sd_y/sd_x`.
	    4. Compute the intercept, `theta_0`, as 
	       `theta_0 = average(yes) - theta_1 * average(x)`* Return `theta_0` and `theta_1`.
    """
    sd_x = x.std()
    sd_y = y.std()
    r = x.corr(y)
    theta_one = r * sd_y /sd_x
    theta_zero = y.mean() - theta_one * x.mean()
    return theta_zero, theta_one

x_cols = ['language_count', 'ap_count', 'pct_stu_safe', 'attendance_rate', 'college_career_rate']
coeff = {}
for col in x_cols:
    coeff[col] = compute_lin_reg(x_train[col], y_train)
    print(f'For {col}, theta_0 = {coeff[col][0]} and theta_1 = {coeff[col][1]}')

# Section 3: Model evaluation
def predict(x, x_theta_0, x_theta_1):
    """Make 1-d linear model prediction on an array of inputs.
    %
     The function returns the predicted values of the dependent variable, `x`, under 
     the linear regression model with y intercept `theta_0` and slope `theta_1`
    
    :param x: array of numeric values representing the independent variable.
    :type x: pandas.Series or numpy.ndarray.
    :param x_theta_0: the y-intercept of the linear regression model.
    :type x_theta_0: float
    :param x_theta_1: the slope of the 1-d linear regression model.
    :type x_theta_1: float
    :returns: array of numeric values with the predictions y = theta_0 + theta_1 * x.
    :rtype: pandas.Series or numpy.ndarray.
    """
    y_pred = []
    for x_val in x:
        y_pred.append(x_theta_0 + x_val*x_theta_1)
    return pd.Series(name=x.name, data=y_pred, index=x.index)

def mse_loss(y_actual, y_estimate):
    """Compute the MSE (mean squared error) loss.    
    
    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: MSE loss between y_actual and y_estimate.
    :rtype: float.
    """
    return np.mean(np.abs(y_actual - y_estimate) ** 2)

def rmse_loss(y_actual, y_estimate):
    """Compute the RMSE loss.
    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: RMSE loss between y_actual and y_estimate.
    :rtype: float.
    """
    return (np.mean((y_actual - y_estimate) ** 2)) ** 0.5

def compute_loss(y_actual, y_estimate, loss_fnc=mse_loss):
    """Compute a user-specified loss.
    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :param loss_fnc: a loss function. Defaults to `mse_loss`.
    :type loss_fnc: function.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: RMSE loss between y_actual and y_estimate.
    :rtype: float.
    """
    return loss_fnc(y_actual, y_estimate)

# loop through all the models, computing train and test loss
y_train_predictions = {}
y_test_predictions = {}
train_losses = {}
test_losses = {}
MIN_LOSS = 1e09
for col in x_cols:
    theta_0, theta_1 = coeff[col]
    y_train_predictions[col] = predict(x_train[col], theta_0, theta_1)
    y_test_predictions[col] = predict(x_test[col], theta_0, theta_1)
    train_losses[col] = compute_loss(y_train, y_train_predictions[col])
    test_losses[col] = compute_loss(y_test, y_test_predictions[col])

# arrange models' train and test losses into a dataframe
losses_df = pd.DataFrame(
    index=x_cols, data={
        "train_loss": train_losses.values(),
        "test_loss": test_losses.values(),
    }
)
print(losses_df)

def graph_data(df, d_col, d_coeff):
    """
    Function to graph the models
    """
    plt.scatter(df[d_col], df['graduation_rate'], label='Actual')
    predict_grad = predict(df[d_col], d_coeff[d_col][0], d_coeff[d_col][1])
    plt.scatter(df[d_col], predict_grad, label='Predicted')
    plt.title(f'{d_col} vs graduation_rate')
    plt.ylabel('graduation_rate')
    plt.xlabel(f'{d_col}')
    plt.legend()
    plt.show()

graph_data(late_df, 'college_career_rate', coeff)
