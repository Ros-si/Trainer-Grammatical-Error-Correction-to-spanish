import os
import sys
import yaml
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
from box import ConfigBox 
from box.exceptions import BoxValueError
import dill

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lee un archivo yaml y devuelve un ConfigBox
    
    Parameters
    ----------
    path_to_yaml : str
        ruta al archivo yaml
        
    Raises
    ----------
        ValueError: si el yaml está vacío
        Exception: cualquier otra excepción
        
    Returns
    ----------
    ConfigBox
        contenido del yaml como objeto
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"Archivo yaml: {path_to_yaml} cargado correctamente")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("El archivo yaml está vacio")
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: list, verbose=True):
    """
    Crea una lista de directorios si no existen

    Parameters
    ----------
    path_to_directories : list
       lista de rutas de directorios
    verbose : bool (optional)
        si se debe loguear la creación
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Creado el directorio en: {path}")


def save_object(file_path, obj):
    """
    Guarda un objeto usando dill 
    Parameters
    ----------
    file_path : str
        ruta donde se guardará el objeto
    obj : object
        objeto a guardar
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Objeto guardado en: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Carga un objeto serializado con dill

    Parameters
    ----------
    file_path : str
        ruta del archivo a cargar
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)