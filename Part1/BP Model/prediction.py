import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from lime import lime_tabular

###########################################

def preprocess_data(company ,start_data, end_date, scaler, n_input=60,n_out = 1, split=0.75,start1=0):
    # Download the data from Yahoo Finance 
    df_og = yf.download(company, start= start_data, 
                    end= end_date, interval="1d")
    
    df_og = df_og.reset_index()

    # Ensure 'Date' is in datetime format
    df_og['Date'] = pd.to_datetime(df_og['Date'])

    # Sort by date to ensure chronological order
    df_og = df_og.sort_values('Date')

    # Calculate the split index
    split_idx = int(len(df_og) * split)

    # Split into training and testing sets
    train_df = df_og.iloc[:split_idx]
    test_df = df_og.iloc[split_idx:]


    # Ensure 'Date' is in datetime format
    df_og['Date'] = pd.to_datetime(df_og['Date'])

    # Sort by date to ensure chronological order
    df_og = df_og.sort_values('Date')

    def calulate_average(row):
        return (row['High'] + row['Low']) / 2
    
    df_og['Average'] = df_og['Close']#df.apply(calulate_average, axis=1)

    # Calculate the split index (3/4 for training, 1/4 for testing)
    split_idx = int(len(df_og) * split)

    # Split into training and testing sets
    train_df = df_og.iloc[:split_idx]
    test_df = df_og.iloc[split_idx:]

    
    
    # Scale the 'Close' prices using MinMaxScaler
    scaler.fit(train_df[['Average']])
    train_df['Average'] = scaler.transform(train_df[['Average']])
    test_df['Average'] = scaler.transform(test_df[['Average']])
    out_let = n_out -1
    def create_sequences(data, date, seq_len=n_input):
        X, y ,d1= [], [] ,[]
        for i in range(seq_len+out_let, len(data)):
            X.append(data[i-(seq_len+out_let):i-out_let-start1])
            if out_let == 0:
                y.append(data[i])
            else:
                y.append(data[i-out_let:i])
            d1.append(date[i])

        return np.array(X), np.array(y), np.array(d1)
    
    # Create sequences for training and testing data
    X_train, y_train, date_train = create_sequences(train_df['Average'].values, train_df['Date'].values, n_input)
    X_test, y_test, date_test = create_sequences(test_df['Average'].values, test_df['Date'].values, n_input)
    
    return X_train, y_train, date_train, X_test, y_test, date_test, scaler , df_og

#########################

def create_model(n_input1 , n_features1, optimizer1='adam', loss1='mean_squared_error',
    metrics1=[MeanSquaredError(), RootMeanSquaredError()]):
    
    # Build the model
    # Input layer
    input_layer = Input(shape=(n_input1, n_features1), name="Input_Layer")  
    
    # GRU layers
    X = GRU(units=64, return_sequences=True, name="GRU_Layer1")(input_layer)  # Return sequences for next GRU
    X = Dropout(0.2, name="Dropout1")(X)  # Prevent overfitting

    X = GRU(units=32, return_sequences=False, name="GRU_Layer2")(X)  # Final GRU output
    X = Dropout(0.2, name="Dropout2")(X)

    # Dense layers
    X = Dense(16, activation='relu', name="Dense_Layer")(X)

    # Output layer
    output_layer = Dense(1, name="Output_Layer")(X)


    # Create the model
    model1 = Model(inputs=input_layer, outputs=output_layer) 

    # Compile the model
    model1.compile(optimizer=optimizer1, loss=loss1 ,    metrics=metrics1 ) 
    
    return model1

###################################

def plot_loss(history, start_epoch=0):
    filds = history.history.keys()
    plt.figure(figsize=(12, 8))
    if 'loss' in filds:
    # Loss
        plt.subplot(2, 2, 1)
        plt.plot(range(start_epoch, len(history.history['loss'])), 
             history.history['loss'][start_epoch:], 
             label='Train Loss')
        if 'val_loss' in filds:
            plt.plot(range(start_epoch, len(history.history['val_loss'])), 
             history.history['val_loss'][start_epoch:], 
             label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    if 'mean_squared_error' in filds:
        # MSE
        plt.subplot(2, 2, 2)
        plt.plot(range(start_epoch, len(history.history['mean_squared_error'])), 
             history.history['mean_squared_error'][start_epoch:], 
             label='Train MSE')
        if 'val_mean_squared_error' in filds:
            plt.plot(range(start_epoch, len(history.history['val_mean_squared_error'])), 
             history.history['val_mean_squared_error'][start_epoch:], 
             label='Validation MSE')
        plt.title('MSE over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()

    if 'root_mean_squared_error' in filds:
    # RMSE
        plt.subplot(2, 2, 3)
        plt.plot(range(start_epoch, len(history.history['root_mean_squared_error'])), 
             history.history['root_mean_squared_error'][start_epoch:], 
             label='Train RMSE')
        if 'val_root_mean_squared_error' in filds:
            plt.plot(range(start_epoch, len(history.history['val_root_mean_squared_error'])), 
             history.history['val_root_mean_squared_error'][start_epoch:], 
             label='Validation RMSE')
        plt.title('RMSE over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()

    plt.tight_layout()
    plt.show()

######################################

def full_dataset(model3 , X_test, y_test,  date_test, scaler1):

    full_actual = np.concatenate([y_test]) #y_train, 
    full_actual = scaler1.inverse_transform(full_actual.reshape(-1, 1)).flatten()

    predicted_test = model3.predict(X_test)
    predicted_prices = scaler1.inverse_transform(predicted_test)
    full_predicted = predicted_prices.flatten()[:]

    date_range = date_test.tolist()
    date_range = pd.to_datetime(date_range)

    df = pd.DataFrame({
        'Date': date_range,
        'Actual': full_actual,
        'Predicted': full_predicted
    })
    return df

#################################################

def calulate_total(df, diplaty1=False,statsistics1=True):
    
    # Calculate the difference: Predicted - Actual
    df['Difference'] = df['Predicted'] - df['Actual']
    if diplaty1:
        # Create a histogram of the differences
        plt.figure(figsize=(6, 3))
        plt.hist(df['Difference'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Histogram of Prediction Differences (Predicted - Actual)')
        plt.xlabel('Difference (Predicted - Actual)')
        plt.ylabel('Number of Days')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    if statsistics1:
        # Optional: Print summary statistics for the differences
        print("Summary statistics for the differences:")
        print(df['Difference'].describe())
    print(f"Total = {df['Difference'].sum():.2f}")

##########################################

def display_trends(orognal_data, predicted_data):
    df_og = orognal_data.reset_index(drop=True)


    # Create a temporary DataFrame for merging without modifying df_og
    temp_df = pd.DataFrame({
        'Date': df_og[('Date', '')],
        'Average': df_og[('Average', '')]
    })

    # Ensure Date is in datetime format in temp_df
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])

    # Ensure df2['Date'] is in datetime format (without modifying df2 directly)
    # Create a copy for merging to avoid altering df2
    df2_merge = predicted_data.copy()
    df2_merge['Date'] = pd.to_datetime(df2_merge['Date'])

    # Merge temp_df and df2_merge on 'Date'
    merged_df = temp_df.merge(df2_merge, on='Date', how='inner')

    # Verify merged_df
    print("Merged DataFrame head:")
    print(merged_df.head())

    # Plot the trend
    plt.figure(figsize=(12, 6))

    # Plot df_og['Average'] for the full trend using MultiIndex
    plt.plot(df_og[('Date', '')], df_og[('Average', '')], label='Original Average', color='blue', alpha=0.5)
    plt.plot(merged_df['Date'], merged_df['Predicted'], label='Predicted', color='red', linestyle='-.')

    # Customize plot
    plt.title('Trend of Average Price with Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12, 6))
    # Plot df2['Actual'] and df2['Predicted'] where dates overlap
    plt.plot(merged_df['Date'], merged_df['Actual'], label='Actual', color='green', linestyle='--')
    plt.plot(merged_df['Date'], merged_df['Predicted'], label='Predicted', color='red', linestyle='-.')

    # Customize plot
    plt.title('Trend of Average Price with Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

###########################################

def rolling_prediction(model, X_train, days, sequence_length=60,start1=0):
    # Initialize predictions list
    predictions = []
    
    # Start with first 60 days from training data
    current_sequence = X_train[start1,:].copy()
    
    # Iterate for the length of y_test
    for _ in range(days):
        # Reshape current_sequence for model input (1, sequence_length, features)
        current_sequence_reshaped = current_sequence.reshape(1, sequence_length)
        
        # Get prediction for next day
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        
        # Append prediction to results
        predictions.append(next_pred[0])
        
        # Update sequence: remove first day, add prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred
        
    # Convert predictions to numpy array
    predictions = np.array(predictions)
    
    return predictions

#####################################

def full_dataset2(model3 , X_test, y_test,  date_test, scaler1):

    full_actual = np.concatenate([y_test]) #y_train, 
    fullactual = scaler1.inverse_transform(full_actual.reshape(-1, 1)).flatten()
    start = 0

    predicted_test =rolling_prediction(model3 , X_test, 
    len(y_test)-start , sequence_length=X_test.shape[1],start1=start)

    predicted_test1 = np.append(y_test[:start],predicted_test )
    predicted_prices = scaler1.inverse_transform(predicted_test1.reshape(-1, 1))

    full_predicted = predicted_prices.flatten()[:]

    date_range = date_test.tolist()
    date_range = pd.to_datetime(date_range)

    df = pd.DataFrame({
        'Date': date_range,
        'Actual': fullactual,
        'Predicted': full_predicted
    })
    return df

#######################################

def explainablity_lime(X_train1, X_test1, model_, input_num, display=True):
    # Reshape the input to match the model's expected input shape
    
# Assume your input has shape (samples, time_steps, features)
# Flatten time dimension for LIME (you can also average or extract specific time step if appropriate)
    X_train_flat = X_train1.reshape((X_train1.shape[0], -1))
    X_test_flat = X_test1.reshape((X_test1.shape[0], -1))
    # Create explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        mode='regression',
        feature_names=[f"Day{i}_{j}" for i in range(X_train1.shape[1]) for j in range(1)],#range(X_train.shape[2])],
        verbose=True
    )
    i = input_num
    exp = explainer.explain_instance(
        data_row=X_test_flat[i],
        predict_fn=lambda x: model_.predict(x.reshape(-1, X_train1.shape[1]))#,1))# X_train.shape[2]))
    )

    # Show explanation
    if display:
        exp.show_in_notebook(show_table=True, show_all=False)
    else:
        # Save the explanation to a file
        with open('lime_explanation.txt', 'w') as f:
            f.write(str(exp.as_list()))
    # exp.show_in_notebook()
    return exp
    # X_train.shape, X_test.shap