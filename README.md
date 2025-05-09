# Experimental design, machine learning, and genetic algorithm to support the production of conductive polymer nanocomposites.

This repository provides the datasets, scripts, and machine-learning models developed for the article “Experimental Design, Machine Learning, and Genetic Algorithm to Support the Production of Conductive Polymer Nanocomposites.” The study integrates materials-informatics methodologies into the sustainable development of advanced materials—approaches that can substantially reduce the extensive resources consumed by conventional trial-and-error methods. Specifically, five polymer matrix types, three conductive fillers, and two processing parameters were evaluated. This work is part of a larger project aimed at fabricating conductive filaments via 3D printing for the bipolar plates of proton-exchange-membrane (PEM) fuel cells.

# Using the best-trained models.

The best Random Forest model is located at doe_ml_ga_conductive_polymers/Machine Learning Models/job_RF_10k/ and is saved in the file "best_random_forest_model.joblib". For more information about this model, please refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

The best MLP Regressor model is located at Machine Learning Models/job_MLP_50K and is saved in the file "best_mlp_model.joblib". For more information about this model, please refer to https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html.

Example of use:

      import pandas as pd
      import joblib
      from sklearn.preprocessing import StandardScaler
      
      # 1. Read the dataset to extract feature names and build a scaler
      df = pd.read_excel("DataFrame_unificado_one_hot.xlsx")
      feature_cols = [c for c in df.columns if c not in ["Resposta(S/cm)", "Unnamed: 0"]]
      
      # 2. Reconstruct the training‐set features (so we can fit the scaler)
      X_train = df[feature_cols].values
      
      # 3. Load (or fit) the scaler
      scaler = StandardScaler().fit(X_train)
      
      # 4. Define your new single‐row sample (raw values)
      new_values = {
          'SW': 1,
          'MW': 0,
          'GR': 0,
          'RPM': 100,
          'Tempo (minutos)': 10,
          'PA66': 0,
          'PC': 0,
          'PEI': 0,
          'PPA': 0,
          'PVDF': 1
      }
      X_new = pd.DataFrame([new_values], columns=feature_cols)
      
      # 5. Apply scaling
      X_new_scaled = scaler.transform(X_new.values)
      
      # 6. Load the trained model and predict
      model = joblib.load("best_mlp_model.joblib")
      y_pred = model.predict(X_new_scaled)
      
      print("Prediction  : "+str( y_pred[0]) + " (S/cm)")



# Using the proposed methodology.

If you are interested in adapting the proposed methodology for another specific project, you should:

  1.Navigate to the experimental design section and modify the code to generate a design suited to your own experiment. It is advisable to consult the PyDOE library documentation at https://pythonhosted.org/pyDOE/.
  
  2.For hyperparameter tuning of your models, we recommend using Optuna (https://optuna.org/). This stage will likely require access to a compute cluster.
  
  3.Adapt our genetic algorithm code to address the specific peculiarities of your problem.
  
  4.To generate informative visualizations of your models’ decision-making processes, we recommend the SHAP library.

# Accessing the article.

The article will be available soon.
