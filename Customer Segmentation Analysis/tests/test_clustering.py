import unittest
import numpy as np
import pandas as pd
import yaml

class TestDataGeneration(unittest.TestCase):

    def setUp(self):

        with open("config.yaml","r") as f:
            self.config=yaml.safe_load(f)

    def test_config_loaded(self):

        self.assertIsNotNone(self.config)

        self.assertIn("data",self.config)
        self.assertIn("clustering",self.config)

    def test_config_values(self):

        self.assertEqual(
            self.config["clustering"]["method"],
            "kmeans"
        )

        self.assertGreater(
            self.config["data"]["n_samples"],
            0
        )

        self.assertGreater(
            self.config["clustering"]["final_k"],
            1
        )

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):

        from data_generation import generate_customer_data

        with open("config.yaml","r") as f:
            self.config=yaml.safe_load(f)

        self.df=generate_customer_data(self.config)

    def test_data_shape(self):

        self.assertEqual(
            len(self.df),
            self.config["data"]["n_samples"]
        )

    def test_data_columns(self):

        cols=[
            "Customer_ID",
            "Age",
            "Annual_Income_K",
            "Spending_Score"
        ]

        for col in cols:
            self.assertIn(col,self.df.columns)

    def test_no_missing_values(self):

        self.assertEqual(
            self.df.isnull().sum().sum(),
            0
        )

    def test_age_range(self):

        min_age=self.config["data"]["features"]["age"]["min"]
        max_age=self.config["data"]["features"]["age"]["max"]

        self.assertTrue(self.df["Age"].min()>=min_age)
        self.assertTrue(self.df["Age"].max()<=max_age)

    def test_income_range(self):

        min_income=self.config["data"]["features"]["income"]["min"]
        max_income=self.config["data"]["features"]["income"]["max"]

        self.assertTrue(
            self.df["Annual_Income_K"].min()>=min_income
        )

        self.assertTrue(
            self.df["Annual_Income_K"].max()<=max_income
        )

class TestClustering(unittest.TestCase):

    def setUp(self):

        from data_generation import generate_customer_data
        from data_preprocessing import DataPreprocessor

        with open("config.yaml","r") as f:
            self.config=yaml.safe_load(f)

        self.df=generate_customer_data(self.config)

        preprocessor=DataPreprocessor(self.config)

        self.X_normalized,self.X=preprocessor.preprocess(self.df)

        self.feature_names=preprocessor.feature_names

    def test_normalized_data_shape(self):

        self.assertEqual(
            self.X_normalized.shape[1],
            len(self.feature_names)
        )

    def test_normalized_data_scaling(self):

        mean=self.X_normalized.mean(axis=0)
        std=self.X_normalized.std(axis=0)

        np.testing.assert_array_almost_equal(
            mean,
            np.zeros_like(mean),
            decimal=1
        )

        np.testing.assert_array_almost_equal(
            std,
            np.ones_like(std),
            decimal=1
        )

    def test_model_creation(self):

        from clustering import CustomerSegmentationModel

        model=CustomerSegmentationModel(self.config)

        self.assertIsNotNone(model)
        self.assertIsNone(model.model)

    def test_model_training(self):

        from clustering import CustomerSegmentationModel

        model=CustomerSegmentationModel(self.config)

        labels=model.train(self.X_normalized)

        self.assertEqual(
            len(labels),
            len(self.X_normalized)
        )

        self.assertIsNotNone(model.model)

    def test_cluster_labels(self):

        from clustering import CustomerSegmentationModel

        model=CustomerSegmentationModel(self.config)

        labels=model.train(self.X_normalized)

        unique=np.unique(labels)

        k=self.config["clustering"]["final_k"]

        self.assertEqual(len(unique),k)

        self.assertTrue(np.all(unique>=0))
        self.assertTrue(np.all(unique<k))

class TestSegmentAnalysis(unittest.TestCase):

    def setUp(self):

        from data_generation import generate_customer_data
        from data_preprocessing import DataPreprocessor
        from clustering import CustomerSegmentationModel,SegmentAnalyzer

        with open("config.yaml","r") as f:
            self.config=yaml.safe_load(f)

        self.df=generate_customer_data(self.config)

        preprocessor=DataPreprocessor(self.config)

        X_normalized,X=preprocessor.preprocess(self.df)

        model=CustomerSegmentationModel(self.config)

        labels=model.train(X_normalized)

        self.analyzer=SegmentAnalyzer(self.df,labels)

    def test_segment_profiles(self):

        profiles=self.analyzer.get_segment_profiles()

        self.assertIsInstance(profiles,pd.DataFrame)

    def test_business_insights(self):

        insights=self.analyzer.generate_business_insights()

        self.assertEqual(
            len(insights),
            self.config["clustering"]["final_k"]
        )

def run_tests():

    unittest.main(
        argv=[""],
        exit=False
    )

if __name__=="__main__":
    run_tests()