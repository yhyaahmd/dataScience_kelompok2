import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)

def run_regression(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding
    cat_cols = X.select_dtypes(include='object').columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, y_pred)


    coef = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "coef": coef
    }
