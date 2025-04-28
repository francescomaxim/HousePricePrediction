# House Price Prediction

## Overview
This project implements a simple linear regression model to predict house prices based on the square footage of living space. The dataset used contains house attributes, and the model is trained on a subset of this data to predict prices.

## Dataset
The dataset (`data.csv`) includes various house features such as square footage, number of bedrooms, bathrooms, and sale prices, among others. For this project, we focus on:
- **Input Feature**: `sqft_living` (square footage of living space)
- **Target Variable**: `price` (sale price of the house)

## Project Structure
- `Untitled.ipynb`: Jupyter Notebook containing the data preprocessing, model training, and visualization code.
- `data.csv`: Dataset with house attributes and prices.

## Dependencies
To run the notebook, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`

You can install them using:
```bash
pip install numpy pandas matplotlib
```

## Methodology
1. **Data Preprocessing**:
   - Load the dataset using `pandas`.
   - Extract `sqft_living` as the input feature and `price` as the target variable.
   - Split the data into training (80%) and validation (20%) sets using a custom `train_validation_split` function.

2. **Model Training**:
   - A linear regression model is trained using a small synthetic dataset (`x_train`, `y_train`) with square footage and corresponding prices.
   - The model parameters (`w_final` for slope and `b_final` for intercept) are used to make predictions.

3. **Visualization**:
   - Scatter plots display the training and validation data points.
   - A linear regression line is plotted to visualize the model's fit on the training data.

4. **Prediction**:
   - The model predicts the price for a house with a given square footage (e.g., 50 units).

## Usage
1. Clone the repository:
   ```bash
   git clone [<repository-url>](https://github.com/francescomaxim/HousePricePrediction)
   ```
2. Navigate to the project directory:
   ```bash
   cd HousePricePrediction
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Untitled.ipynb
   ```
4. Run the cells in the notebook to preprocess the data, train the model, and visualize results.

## Results
- The model provides a simple linear relationship between square footage and house price.
- Example prediction: For a house with 50 units of square footage, the predicted price is approximately $12,816.64 (based on the provided model parameters).
- Visualizations show the training data points and the fitted linear regression line.

## Limitations
- The model uses a small synthetic dataset for training, which may not generalize well to the full dataset.
- Only one feature (`sqft_living`) is considered, ignoring other potentially relevant features like bedrooms or location.
- The linear regression assumption may not capture complex relationships in real-world housing data.

## Future Improvements
- Train the model on the full dataset using gradient descent or a library like `scikit-learn`.
- Incorporate additional features (e.g., `bedrooms`, `bathrooms`, `yr_built`) for better predictions.
- Evaluate the model using metrics like Mean Squared Error (MSE) or R² score.
- Explore non-linear models or feature engineering to improve accuracy.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
