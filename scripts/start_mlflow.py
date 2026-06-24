import socket
import subprocess
import time


def is_mlflow_running(
    host="127.0.0.1",
    port=5000
):
    try:
        with socket.create_connection(
            (host, port),
            timeout=1
        ):
            return True
    except Exception:
        return False


def start_mlflow():

    if is_mlflow_running():
        print("MLflow já está em execução.")
        return

    print("Iniciando MLflow...")

    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--default-artifact-root",
            "./mlartifacts",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
    )

    for _ in range(15):

        if is_mlflow_running():
            print("MLflow iniciado com sucesso.")
            return

        time.sleep(1)

    print("Não foi possível iniciar o MLflow.")