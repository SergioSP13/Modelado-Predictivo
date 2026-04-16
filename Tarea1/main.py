import pandas as pd
import sys
import joblib


def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Clasifica un pingüino basándose en reglas diseñadas por un humano.
    """
    
    if flipper_length_mm > 210:
        return "Gentoo"
    elif flipper_length_mm <= 202:
        if bill_length_mm < 40.5:
            return "Adelie"
        elif bill_length_mm > 46:
            return "Chinstrap"
        else:
            if bill_depth_mm < 17.2:
                return "Chinstrap"
            else: 
                return "Adelie"
    else:
        if bill_depth_mm < 17:
            return "Gentoo"
        else:
            if bill_length_mm < 47:
                return "Adelie"
            else:
                return "Chinstrap" 


def main():

    if len(sys.argv) < 2:
        print("Uso: python clasificador_pinguinos.py dataset.csv")
        sys.exit(1)

    archivo = sys.argv[1]

    # Cargar dataset
    df = pd.read_csv(archivo)

    # Variables de entrada
    X = df[['bill_length_mm',
            'bill_depth_mm',
            'flipper_length_mm',
            'body_mass_g']]

    # Cargar modelo entrenado
    modelo = joblib.load("modelo_pinguinos.pkl")

    # Predicción del modelo
    pred_modelo = modelo.predict(X)

    # Predicción del clasificador humano
    pred_humano = X.apply(
        lambda row: clasificador_humano(
            row['bill_length_mm'],
            row['bill_depth_mm'],
            row['flipper_length_mm'],
            row['body_mass_g']
        ),
        axis=1
    )

    # Crear dataframe solo con las predicciones
    resultado = pd.DataFrame({
        "pred_modelo": pred_modelo,
        "pred_humano": pred_humano
    })

    # Guardar salida
    salida = "predicciones.csv"
    resultado.to_csv(salida, index=False)

    print("Archivo generado:", salida)
    print(resultado.head())


if __name__ == "__main__":
    main()