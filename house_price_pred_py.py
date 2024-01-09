{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "mount_file_id": "1aWvX6V-Bg0s92lGDVYUXm8UwXL8KmmOI",
      "authorship_tag": "ABX9TyMEWzSO66JKiDiMvYcw2pLN",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/amsalugetasew/zero_day/blob/main/house_price_pred_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit pandas scikit-learn"
      ],
      "metadata": {
        "id": "8MAvepFBCt5b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_squared_error"
      ],
      "metadata": {
        "id": "WyDP9FtCCzPE"
      },
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load your dataset (replace 'your_data.csv' with your dataset)\n",
        "data = pd.read_csv('/content/drive/MyDrive/House_Price_Prediction/housing_price_dataset - housing_price_dataset.csv.csv')"
      ],
      "metadata": {
        "id": "5_rQ2DxpC969"
      },
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Assume the dataset has features (X) and target variable (y)\n",
        "# Adjust this based on your actual dataset\n",
        "X = data.drop('Price', axis=1)  # Features\n",
        "y = data['Price']  # Target variable"
      ],
      "metadata": {
        "id": "5XHgbWuSDP_b"
      },
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X.isnull().value_counts"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WHiIZIbGIL3Y",
        "outputId": "50040d02-74ef-47f8-dafb-804dc65fb1db"
      },
      "execution_count": 70,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<bound method DataFrame.value_counts of        SquareFeet  Bedrooms  Bathrooms  Neighborhood  YearBuilt\n",
              "0           False     False      False         False      False\n",
              "1           False     False      False         False      False\n",
              "2           False     False      False         False      False\n",
              "3           False     False      False         False      False\n",
              "4           False     False      False         False      False\n",
              "...           ...       ...        ...           ...        ...\n",
              "49995       False     False      False         False      False\n",
              "49996       False     False      False         False      False\n",
              "49997       False     False      False         False      False\n",
              "49998       False     False      False         False      False\n",
              "49999       False     False      False         False      False\n",
              "\n",
              "[50000 rows x 5 columns]>"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "GBZMNzqWJHaY"
      },
      "execution_count": 71,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 72,
      "metadata": {
        "id": "MafiFnTNCoBx"
      },
      "outputs": [],
      "source": [
        "X_train['Neighborhood']= X_train['Neighborhood'].replace({'Rural': 1.0, 'Urban': 2.0, 'Suburb': 3.0})\n",
        "X_test['Neighborhood']= X_test['Neighborhood'].replace({'Rural': 1.0, 'Urban': 2.0, 'Suburb': 3.0})\n",
        "X['Neighborhood']= X_train['Neighborhood'].replace({'Rural': 1.0, 'Urban': 2.0, 'Suburb': 3.0})\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Assuming 'df' is your DataFrame\n",
        "X.dropna(inplace=True)"
      ],
      "metadata": {
        "id": "54ldkagkIxSg"
      },
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Train the model (replace RandomForestRegressor with your actual model)\n",
        "model = RandomForestRegressor()\n",
        "model.fit(X_train, y_train)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "wIGiTtj2HAFR",
        "outputId": "470b2182-b453-4c08-b2b9-fc3c461aa3a2"
      },
      "execution_count": 74,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestRegressor()"
            ],
            "text/html": [
              "<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-7\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" checked><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 74
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Make predictions\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "\n",
        "# Streamlit app\n",
        "st.title('House Price Prediction App')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JLMuMXjpDedn",
        "outputId": "99f83e1d-1787-4f1b-a23b-c4ec95d91851"
      },
      "execution_count": 75,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Sidebar for user input\n",
        "st.sidebar.header('User Input Features')\n",
        "# Assume features are 'Feature1', 'Feature2', etc.\n",
        "feature1 = st.sidebar.slider('SquareFeet', float(X['SquareFeet'].min()), float(X['SquareFeet'].max()), float(X['SquareFeet'].mean()))\n",
        "feature2 = st.sidebar.slider('Bedrooms', float(X['Bedrooms'].min()), float(X['Bedrooms'].max()), float(X['Bedrooms'].mean()))\n",
        "feature3 = st.sidebar.slider('Bathrooms', float(X['Bathrooms'].min()), float(X['Bathrooms'].max()), float(X['Bathrooms'].mean()))\n",
        "feature4 = st.sidebar.slider('Neighborhood', float(X['Neighborhood'].min()), float(X['Neighborhood'].max()), float(X['Neighborhood'].mean()))\n",
        "feature5 = st.sidebar.slider('YearBuilt', float(X['YearBuilt'].min()), float(X['YearBuilt'].max()), float(X['YearBuilt'].mean()))\n",
        "\n",
        "# Create a DataFrame for user input\n",
        "user_input = pd.DataFrame({'SquareFeet': [feature1], 'Bedrooms': [feature2], 'Bathrooms': [feature3], 'Neighborhood': [feature4],'YearBuilt': [feature5]})\n",
        "\n"
      ],
      "metadata": {
        "id": "co8YIBAkDjBD"
      },
      "execution_count": 76,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Make predictions based on user input\n",
        "prediction = model.predict(user_input)\n",
        "\n",
        "# Display prediction\n",
        "st.subheader('Predicted House Price:')\n",
        "st.write(f\"${prediction[0]:,.2f}\")\n",
        "\n",
        "# Display model evaluation metric\n",
        "st.subheader('Model Evaluation (Mean Squared Error):')\n",
        "st.write(f'MSE: {mse:.2f}')"
      ],
      "metadata": {
        "id": "0nZZ7IuwDuMA"
      },
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run house_price_pred.py &"
      ],
      "metadata": {
        "id": "yHX-9R6ILHJl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run htttps://hub.com/amsalugetasew/python_projects/blob/main/house_price_pred_py.py"
      ],
      "metadata": {
        "id": "qRnO99XgPB4k"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}