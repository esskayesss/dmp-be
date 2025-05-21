from enum import Enum
import pandas as pd
from predict.knn import knn_predict
from predict.knn4p import knn_predict as knn_predict4p

from lib import preprocess_user_query
from lib4p import preprocess_user_query as preprocess_user_query_4p


# Enum for Shape
class Shape(Enum):
    ASSCHER = "asscher"
    CUSHION = "cushion"
    CUSHION_MODIFIED = "cushion_modified"
    EMERALD = "emerald"
    HEART = "heart"
    MARQUISE = "marquise"
    OVAL = "oval"
    PEAR = "pear"
    PRINCESS = "princess"
    RADIANT = "radiant"
    ROUND = "round"
    SQUARE_RADIANT = "square_radiant"


# Enum for Cut
class Cut(Enum):
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    EXCELLENT = "Excellent"
    IDEAL = "Ideal"


# Enum for Color
class Color(Enum):
    M = "M"
    L = "L"
    K = "K"
    J = "J"
    I = "I"
    H = "H"
    G = "G"
    F = "F"
    E = "E"
    D = "D"


# Enum for Clarity
class Clarity(Enum):
    I1 = "I1"
    SI2 = "SI2"
    SI1 = "SI1"
    VS2 = "VS2"
    VS1 = "VS1"
    VVS2 = "VVS2"
    VVS1 = "VVS1"
    IF = "IF"
    FL = "FL"


# Enum for Fluorescence
class Fluorescence(Enum):
    STRONG = "Strong"
    STRONG_BLUE = "Strong Blue"
    MEDIUM = "Medium"
    MEDIUM_BLUE = "Medium Blue"
    SLIGHT = "Slight"
    FAINT = "Faint"
    NEGLIGIBLE = "Negligible"
    NONE = "None"


class Diamond:
    def __init__(
        self,
        carat: float,
        shape: str,
        cut: str,
        color: str,
        clarity: str,
        fluorescence: str,
        x: float,
        y: float,
        z: float,
        depth: float,
        table: float,
    ):
        # Validate numerical features
        numerical_attrs = {
            "carat": carat,
            "x": x,
            "y": y,
            "z": z,
            "depth": depth,
            "table": table,
        }
        for attr_name, attr_value in numerical_attrs.items():
            if not isinstance(attr_value, (int, float)) or attr_value <= 0:
                raise ValueError(
                    f"{attr_name} must be a positive number, got {attr_value}"
                )

        # Validate categorical features using Enums
        try:
            self.shape = Shape(shape).value
        except ValueError:
            raise ValueError(
                f"Invalid shape: {shape}. Must be one of {[e.value for e in Shape]}"
            )

        try:
            self.cut = Cut(cut).value
        except ValueError:
            raise ValueError(
                f"Invalid cut: {cut}. Must be one of {[e.value for e in Cut]}"
            )

        try:
            self.color = Color(color).value
        except ValueError:
            raise ValueError(
                f"Invalid color: {color}. Must be one of {[e.value for e in Color]}"
            )

        try:
            self.clarity = Clarity(clarity).value
        except ValueError:
            raise ValueError(
                f"Invalid clarity: {clarity}. Must be one of {[e.value for e in Clarity]}"
            )

        try:
            self.fluorescence = Fluorescence(fluorescence).value
        except ValueError:
            raise ValueError(
                f"Invalid fluorescence: {fluorescence}. Must be one of {[e.value for e in Fluorescence]}"
            )

        # Set numerical attributes
        self.carat = float(carat)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.depth = float(depth)
        self.table = float(table)

    def predict_price(self, k=12) -> float:
        user_df = pd.DataFrame([self.to_dict()])
        user_features = preprocess_user_query(user_df)
        return knn_predict(user_features, k=k)

    def to_dict(self):
        """Convert the Diamond object to a dictionary for preprocessing."""
        return {
            "carat": self.carat,
            "shape": self.shape,
            "cut": self.cut,
            "color": self.color,
            "clarity": self.clarity,
            "fluorescence": self.fluorescence,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "depth": self.depth,
            "table": self.table,
        }

    def __repr__(self):
        return (
            f"Diamond(carat={self.carat}, shape={self.shape}, cut={self.cut}, "
            f"color={self.color}, clarity={self.clarity}, fluorescence={self.fluorescence}, "
            f"x={self.x}, y={self.y}, z={self.z}, depth={self.depth}, table={self.table})"
        )

    def __hash__(self):
        """Generate a hash based on the diamond's attributes."""
        return hash((
            self.carat,
            self.shape,
            self.cut,
            self.color,
            self.clarity,
            self.fluorescence,
            self.x,
            self.y,
            self.z,
            self.depth,
            self.table,
        ))

    def __eq__(self, other):
        """Check equality based on the diamond's attributes."""
        if not isinstance(other, Diamond):
            return False
        return (
            self.carat == other.carat and
            self.shape == other.shape and
            self.cut == other.cut and
            self.color == other.color and
            self.clarity == other.clarity and
            self.fluorescence == other.fluorescence and
            self.x == other.x and
            self.y == other.y and
            self.z == other.z and
            self.depth == other.depth and
            self.table == other.table
        )



class Diamond4P:
    def __init__(
        self,
        carat: float,
        shape: str,
        color: str,
        clarity: str,
    ):
        # Validate numerical features
        numerical_attrs = {
            "carat": carat,
        }
        for attr_name, attr_value in numerical_attrs.items():
            if not isinstance(attr_value, (int, float)) or attr_value <= 0:
                raise ValueError(
                    f"{attr_name} must be a positive number, got {attr_value}"
                )

        try:
            self.shape = Shape(shape).value
        except ValueError:
            raise ValueError(
                f"Invalid shape: {shape}. Must be one of {[e.value for e in Shape]}"
            )

        try:
            self.color = Color(color).value
        except ValueError:
            raise ValueError(
                f"Invalid color: {color}. Must be one of {[e.value for e in Color]}"
            )

        try:
            self.clarity = Clarity(clarity).value
        except ValueError:
            raise ValueError(
                f"Invalid clarity: {clarity}. Must be one of {[e.value for e in Clarity]}"
            )

        # Set numerical attributes
        self.carat = float(carat)

    def predict_price(self, k=12) -> float:
        user_df = pd.DataFrame([self.to_dict()])
        user_features = preprocess_user_query_4p(user_df)
        return knn_predict4p(user_features, k=k)

    def to_dict(self):
        """Convert the Diamond object to a dictionary for preprocessing."""
        return {
            "carat": self.carat,
            "shape": self.shape,
            "color": self.color,
            "clarity": self.clarity,
        }

    def __repr__(self):
        return (
            f"Diamond(carat={self.carat}, shape={self.shape}, cut={self.cut}, "
            f"color={self.color}, clarity={self.clarity}, fluorescence={self.fluorescence}, "
            f"x={self.x}, y={self.y}, z={self.z}, depth={self.depth}, table={self.table})"
        )

    def __hash__(self):
        """Generate a hash based on the diamond's attributes."""
        return hash(
            (
                self.carat,
                self.shape,
                self.color,
                self.clarity,
            )
        )

    def __eq__(self, other):
        """Check equality based on the diamond's attributes."""
        if not isinstance(other, Diamond):
            return False
        return (
            self.carat == other.carat
            and self.shape == other.shape
            and self.color == other.color
            and self.clarity == other.clarity
        )
