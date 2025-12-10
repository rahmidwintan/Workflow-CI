import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="C:\Rahmi\kuliah\SEMESTER 7\ASAH\SMSML_Rahmi Dwi Intan\Eksperimen_SML_Rahmi-Dwi-Intan\Preprocessing\StudentsPerformance_raw\StudentsPerformance.csv")
    parser.add_argument("--output", required=True, help="C:\Rahmi\kuliah\SEMESTER 7\ASAH\SMSML_Rahmi Dwi Intan\Eksperimen_SML_Rahmi-Dwi-Intan\Preprocessing\StudentsPerformance_preprocessing\dataset_preprocessed.csv")
    args = parser.parse_args()

    raw_file = args.input
    output_file = args.output

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load dataset
    data = pd.read_csv("C:\Rahmi\kuliah\SEMESTER 7\ASAH\SMSML_Rahmi Dwi Intan\Eksperimen_SML_Rahmi-Dwi-Intan\Preprocessing\StudentsPerformance_raw\StudentsPerformance.csv")

    # Bersihkan nama kolom
    data.columns = [col.strip() for col in data.columns]

    # Kolom numerik & kategori
    num_cols = ['writing score', 'math score', 'reading score']
    cat_cols = [
        'parental level of education',
        'test preparation course',
        'race/ethnicity',
        'lunch',
        'gender'
    ]

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    transformer = ColumnTransformer([
        ('num', scaler, num_cols),
        ('cat', encoder, cat_cols)
    ])

    processed = transformer.fit_transform(data)

    encoded_features = transformer.named_transformers_['cat'].get_feature_names_out(cat_cols)
    final_cols = num_cols + list(encoded_features)

    df = pd.DataFrame(processed, columns=final_cols)

    df.to_csv(output_file, index=False)
    print(f"Dataset preprocessed saved to: {output_file}")


if __name__ == "__main__":
    main()
