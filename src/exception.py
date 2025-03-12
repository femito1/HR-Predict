import sys
from src.logger import logging
# This is custom Error/Exception handling

def error_msg_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "The error occured in Py Script name [{0}], Line Number [{1}], Error message [{2}]".format(
        filename,exc_tb.tb_lineno,str(error))
        # The above line is filling in the placeholders for 0, 1, 2
        
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error_message, error_detail=error_detail)
        # err_msg_details takes 2 args and return val is a singular error messsage

    def __str__(self):
        return self.error_message