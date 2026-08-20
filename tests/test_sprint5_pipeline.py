import importlib.util
from pathlib import Path
import unittest

import pandas as pd


PIPELINE = Path(__file__).parents[1] / "competitions" / "smartphone-addiction" / "code" / "sprint5_omralinov" / "pipeline.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("sprint5_pipeline", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sprint5PipelineTests(unittest.TestCase):
    def test_catboost_smoke_converts_all_categorical_missing_values(self):
        pipeline = load_pipeline()
        data_dir = PIPELINE.parents[2] / "data"
        train = pd.read_csv(data_dir / "train.csv").sample(400, random_state=20260820).reset_index(drop=True)
        test = pd.read_csv(data_dir / "test.csv").sample(120, random_state=20260820).reset_index(drop=True)
        train, test = pipeline.prepare_data(train, test)

        oof, test_pred = pipeline.train_catboost(train, test, n_folds=2, seed=20260820)

        self.assertEqual(len(oof), len(train))
        self.assertEqual(len(test_pred), len(test))
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())
        self.assertTrue(((test_pred >= 0) & (test_pred <= 1)).all())

    def test_kaggle_kernel_prepare_data_handles_categorical_columns(self):
        kernel_path = PIPELINE.parent / "sprint5_kaggle_kernel.py"
        spec = importlib.util.spec_from_file_location("sprint5_kernel", kernel_path)
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        data_dir = PIPELINE.parents[2] / "data"
        train = pd.read_csv(data_dir / "train.csv").sample(100, random_state=20260820).reset_index(drop=True)
        test = pd.read_csv(data_dir / "test.csv").sample(40, random_state=20260820).reset_index(drop=True)

        train, test = kernel.prepare_data(train, test)

        self.assertIn("gender", train.columns)
        self.assertFalse(train["gender"].isna().any())
        self.assertFalse(test["gender"].isna().any())


if __name__ == "__main__":
    unittest.main()
