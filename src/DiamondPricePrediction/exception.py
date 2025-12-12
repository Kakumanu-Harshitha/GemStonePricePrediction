import sys

class customexception(Exception):
    """Custom exception class that captures error message, filename, and line number."""

    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        
        # Extract traceback information
        _, _, exc_tb = error_details.exc_info()
        print(exc_tb)  # For debugging (optional)

        # Capture line number and file where error occurred
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """Format the exception output message."""
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )


if __name__ == "__main__":
    try:
        a = 1 / 0  # Force an error for testing
    except Exception as e:
        raise customexception(e, sys)
