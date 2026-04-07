import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    """
    Función para formatear el mensaje de error con detalles del archivo y la línea donde ocurrió el error
    """
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script: [{0}] at line number: [{1}] error message: [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    """
    Clase personalizada de excepción. Se utiliza para capturar y formatear los mensajes de error con detalles específicos del contexto donde ocurrió la excepción.
    """

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
