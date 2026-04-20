from src.data_loader import load_data
from src.outlier_detection import remove_outliers
from src.preprocessing import iqr_capping
from src.aqi_calculation import calculate_aqi
from src.model_training import train_model

def main():

    df = load_data()
    df_clean = remove_outliers(df)
    df_capped = iqr_capping(df_clean)
    df_final, y = calculate_aqi(df_capped)

    train_model(df_final, y)


if __name__ == "__main__":
    main()