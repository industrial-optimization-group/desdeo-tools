from typing import Callable, List, Union

import pandas as pd

from desdeo_tools.interaction.validators import validate_ref_point_with_ideal_and_nadir
from desdeo_tools.utils.frozen import FrozenClass


class RequestError(Exception):
    """Raised when an error related to the Request class is encountered.
    """


class BaseRequest(FrozenClass):
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
        # Attributes
        self._request_type: str = request_type
        self._interaction_priority: str = interaction_priority  # Example: one of:
        self._request_id: int = request_id  # Some random number as id
        self._content = content
        self._response = None
        #  Freezing this class
        self._freeze()

    @property
    def request_type(self):
        return self._request_type

    @property
    def interaction_priority(self):
        return self._interaction_priority

    @property
    def request_id(self):
        return self._request_id

    @property
    def content(self):
        return self._content

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self):
        # Code to validate the response
        return


class PrintRequest(BaseRequest):
    def __init__(self, message: Union[str, List[str]], request_id=None):
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Some elements of the list are not strings"
                )
                raise RequestError(msg)
        super().__init__(
            request_type="print",
            interaction_priority="no_interaction",
            content=message,
            request_id=request_id,
        )


class SimplePlotRequest(BaseRequest):
    def __init__(
        self,
        data: pd.DataFrame,
        message: Union[str, List[str]],
        dimensions_data: pd.DataFrame = None,
        chart_title: str = None,
        request_id=None,
    ):
        acceptable_dimensions_data_indices = [
            "minimize",  # 1 if minimized, -1 if maximized
            "ideal",
            "nadir",
        ]
        if not isinstance(data, pd.DataFrame):
            msg = (
                f"Provided data to be plotted should be in a pandas dataframe, with"
                f"columns names being the same as objective names.\n"
                f"Provided data is of type: {type(data)}"
            )
            raise RequestError(msg)
        if not isinstance(dimensions_data, (pd.DataFrame, type(None))):
            msg = (
                f"Dimensional data should be in a pandas dataframe.\n"
                f"Provided data is of type: {type(dimensions_data)}"
            )
            raise RequestError(msg)
        if not all(data.columns == dimensions_data.columns):
            msg = (
                f"Mismatch in column names of data and dimensions_data.\n"
                f"Column names in data: {data.columns}"
                f"Column names in dimensions_data: {dimensions_data.columns}"
            )
            raise RequestError(msg)
        rouge_indices = [
            index
            for index in dimensions_data.index
            if index not in acceptable_dimensions_data_indices
        ]
        if rouge_indices:
            msg = (
                f"dimensions_data should only contain the following indices:\n"
                f"{acceptable_dimensions_data_indices}\n"
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
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "data": data,
            "dimensions_data": dimensions_data,
            "chart_title": chart_title,
            "message": message,
        }
        super().__init__(
            request_type="simple_plot",
            interaction_priority="no_interaction",
            content=content,
            request_id=request_id,
        )


class ReferencePointPreference(BaseRequest):
    def __init__(
        self,
        dimensions_data: pd.DataFrame,
        message: str = None,
        interaction_priority: str = "required",
        preference_validator: Callable = None,
        request_id: int = None,
    ):
        if message is None:
            message = (
                f"Please provide a reference point better than:\n"
                f"{dimensions_data.loc['nadir'].values.tolist()},\n"
                f"but worse than:\n"
                f"{dimensions_data.loc['ideal'].values.tolist()}"
            )
        if preference_validator is None:
            preference_validator = validate_ref_point_with_ideal_and_nadir
        acceptable_dimensions_data_indices = [
            "minimize",  # 1 if minimized, -1 if maximized
            "ideal",
            "nadir",
        ]
        if not isinstance(dimensions_data, (pd.DataFrame, type(None))):
            msg = (
                f"Dimensional data should be in a pandas dataframe.\n"
                f"Provided data is of type: {type(dimensions_data)}"
            )
            raise RequestError(msg)
        rouge_indices = [
            index
            for index in dimensions_data.index
            if index not in acceptable_dimensions_data_indices
        ]
        if rouge_indices:
            msg = (
                f"dimensions_data should only contain the following indices:\n"
                f"{acceptable_dimensions_data_indices}\n"
                f"The dataframe provided contains the following unsupported indices:\n"
                f"{rouge_indices}"
            )
            raise RequestError(msg)
        if not isinstance(message, str):
            if not isinstance(message, list):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Message provided is of type: {type(message)}"
                )
                raise RequestError(msg)
            elif not all(isinstance(x, str) for x in message):
                msg = (
                    f"Message/s to be printed should be string or list of strings"
                    f"Some elements of the list are not strings"
                )
                raise RequestError(msg)
        content = {
            "dimensions_data": dimensions_data,
            "message": message,
            "validator": preference_validator,
        }
        super().__init__(
            request_type="reference_point_preference",
            interaction_priority=interaction_priority,
            content=content,
            request_id=request_id,
        )

    @BaseRequest.response.setter
    def response(self, value):
        if not isinstance(value, pd.DataFrame):
            msg = "Reference should be provided in a pandas dataframe format"
            raise RequestError(msg)
        self.content["validator"](
            reference_point=value, dimensions_data=self.content["dimensions_data"]
        )
        self._response = value
