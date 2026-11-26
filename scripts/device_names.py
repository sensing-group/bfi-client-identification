"""Names of the six labeled devices, one source for every script and figure.

Three vocabularies name the same devices:
  archive label  the label of xres_common.DEVICE_GT_OUI and of the release's
                 device_labels.csv, e.g. "Apple TV (living room)";
  short key      the key of the evaluation CSVs and JSONs, e.g. "AppleTV-LR";
  Table 2 name   the name printed in Table 2 of the paper and on every figure,
                 e.g. "Apple TV, LR".
"""

SHORT_OF_LABEL = {
    "Apple TV (living room)": "AppleTV-LR", "Apple TV (bedroom)": "AppleTV-BR",
    "Samsung TV": "SamsungTV", "LG TV": "LGTV",
    "Laptop": "Laptop", "iRobot vacuum": "Vacuum",
}
# Device names as printed in Table 2 of the paper, by archive label.
TABLE2_NAME = {"Apple TV (living room)": "Apple TV, LR", "Apple TV (bedroom)": "Apple TV, BR",
               "LG TV": "LG TV", "Laptop": "Laptop",
               "iRobot vacuum": "Robot vacuum", "Samsung TV": "Samsung TV"}
# The same names by short key.
TABLE2_NAME_OF_SHORT = {SHORT_OF_LABEL[lab]: name for lab, name in TABLE2_NAME.items()}
assert set(TABLE2_NAME) == set(SHORT_OF_LABEL)
# Row order of Table 2, used top to bottom by every per-device figure.
TABLE2_ORDER = tuple(SHORT_OF_LABEL[lab] for lab in TABLE2_NAME)
