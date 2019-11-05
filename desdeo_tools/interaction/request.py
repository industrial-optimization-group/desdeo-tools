from typing import List, Union

import pandas as pd


class RequestError(Exception):
    """Raised when an error related to the Request class is encountered.
    """


class BaseRequest:
    def __init__(
        self,
        request_type: str,
        interaction_priority: str,
        content=None,
        request_id: int = None,
    ):
        acceptable_types = [
            "print",
            "simple_plot",
            "reference_point_preference",
            "classification_preference",
        ]
        priority_types = ["no_interaction", "not_required", "recommended", "required"]
        if request_type not in acceptable_types:
            msg = f"Request type should be one of {acceptable_types}"
            raise RequestError(msg)
        if interaction_priority not in priority_types:
            msg = f"Request priority should be one of {priority_types}"
            raise RequestError(msg)
        if not isinstance(request_id, (int, type(None))):
            msg = "Request id should be int or None"
            raise RequestError(msg)
        self.request_type: str = request_type
        self.interaction_priority: str = interaction_priority  # Example: one of:
        self.request_id: int = request_id  # Some random number as id
        self.content = content
        self.response = None


class PrintRequest(BaseRequest):
    def __init__(self, printmessage: Union[str, List[str]], request_id=None):
        if not isinstance(printmessage, str):
            if not isinstance(printmessage, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(printmessage)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in printmessage):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Some elements of the list are not strings"
                )
                raise RequestError(msg)
        super.__init__(
            request_type="print",
            interaction_priority="no_interaction",
            content=print,
            request_id=request_id,
        )


class SimplePlotRequest(BaseRequest):
    def __init__(
        self,
        data: pd.DataFrame,
        dimension_data: pd.DataFrame = None,
        chart_title: str = None,
        printmessage: Union[str, List[str]] = None,
        request_id=None,
    ):
        acceptable_dimensional_data_indices = ["lower_limit", "upper_limit", "maximize"]
        if not isinstance(data, pd.DataFrame):
            msg = (
                f"Provided data to be plotted should be in a pandas dataframe, with"
                f"columns names being the same as objective names.\n"
                f"Provided data is of type: {type(data)}"
            )
            raise RequestError(msg)
        if not isinstance(dimension_data, (pd.DataFrame, type(None))):
            msg = (
                f"Dimensional data should be in a pandas dataframe.\n"
                f"Provided data is of type: {type(dimension_data)}"
            )
            raise RequestError(msg)
        if not all(data.columns == dimension_data.columns):
            msg = (
                f"Mismatch in column names of data and dimensional_data.\n"
                f"Column names in data: {data.columns}"
                f"Column names in dimensional_data: {dimension_data.columns}"
            )
            raise RequestError(msg)
        rouge_indices = [
            index
            for index in dimension_data.index
            if index not in acceptable_dimensional_data_indices
        ]
        if rouge_indices:
            msg = (
                f"dimensional_data should only contain the following indices:\n"
                f"{acceptable_dimensional_data_indices}\n"
                f"The dataframe provided contains the following unsupported indices:\n"
                f"{rouge_indices}"
            )
            raise RequestError(msg)
        if not isinstance(chart_title, (str, type(None))):
            msg = (
                f"Chart title should be a string. Provided chart type is:"
                f"{type(chart_title)}"
            )
            raise RequestError(msg)
        if not isinstance(printmessage, str):
            if not isinstance(printmessage, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(printmessage)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in printmessage):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "data": data,
            "dimensional_data": dimension_data,
            "chart_title": chart_title,
            "print_message": printmessage,
        }
        super.__init__(
            request_type="simple_plot",
            interaction_priority="no_interaction",
            content=content,
            request_id=request_id,
        )
