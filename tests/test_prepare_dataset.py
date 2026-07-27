import pandas as pd

from scripts.prepare_dataset import HIGHER_CONCERN_CLASSES, assign_labels


def test_all_concerning_categories_are_positive():
    rows = []
    columns = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
    for index, diagnosis in enumerate(columns):
        row = {column: 0.0 for column in columns}
        row.update({"lesion_id": f"lesion-{index}", diagnosis: 1.0})
        rows.append(row)
    labeled = assign_labels(pd.DataFrame(rows))
    expected = labeled["diagnosis"].isin(HIGHER_CONCERN_CLASSES).astype(int)
    assert labeled["label"].tolist() == expected.tolist()
