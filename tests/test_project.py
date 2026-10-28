import unittest

import cv2
import numpy as np

import dip_pipeline
from disease_info import get_disease_info
from trainer import TrainingError, _image_group, split_dataset, validate_dataset


class DiseaseInformationTests(unittest.TestCase):
    def test_all_dataset_classes_have_information(self):
        labels = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Pepper__bell___Bacterial_spot",
            "Pepper__bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            "Tomato_healthy",
        ]
        for label in labels:
            with self.subTest(label=label):
                info = get_disease_info(label)
                self.assertTrue(info["known"])
                self.assertTrue(info["symptoms"])
                self.assertTrue(info["management"])


class ValidationSplitTests(unittest.TestCase):
    def test_training_requires_a_healthy_class(self):
        data = {"Apple_scab": ["one.jpg", "two.jpg"], "Apple_black_rot": ["three.jpg", "four.jpg"]}
        with self.assertRaisesRegex(TrainingError, "Healthy"):
            validate_dataset(data)

    def test_training_accepts_a_healthy_class(self):
        data = {"Healthy": ["one.jpg", "two.jpg"], "Apple_scab": ["three.jpg", "four.jpg"]}
        self.assertEqual(validate_dataset(data), ["Apple_scab", "Healthy"])

    def test_augmented_siblings_stay_in_same_split(self):
        data = {
            "Apple_scab": [
                "/data/Apple_scab/11111111-1111-4111-8111-111111111111___leaf.jpg",
                "/data/Apple_scab/11111111-1111-4111-8111-111111111111___leaf_90deg.jpg",
                "/data/Apple_scab/11111111-1111-4111-8111-111111111111___leaf_new30degFlipLR.jpg",
                "/data/Apple_scab/22222222-2222-4222-8222-222222222222___leaf.jpg",
                "/data/Apple_scab/33333333-3333-4333-8333-333333333333___leaf.jpg",
            ],
            "Corn_rust": [
                "/data/Corn_rust/RS_Rust 1566.JPG",
                "/data/Corn_rust/RS_Rust 1566_flipLR.JPG",
                "/data/Corn_rust/RS_Rust 1567.JPG",
                "/data/Corn_rust/RS_Rust 1568.JPG",
            ],
        }
        train, validation = split_dataset(data, seed=7)
        train_paths = {path for path, _ in train}
        validation_paths = {path for path, _ in validation}
        self.assertFalse(train_paths & validation_paths)

        for paths in data.values():
            groups = {_image_group(path) for path in paths}
            for group in groups:
                group_paths = {path for path in paths if _image_group(path) == group}
                self.assertTrue(group_paths <= train_paths or group_paths <= validation_paths)

        self.assertEqual({label for _, label in train}, {0, 1})
        self.assertEqual({label for _, label in validation}, {0, 1})

    def test_split_is_deterministic(self):
        data = {"class": [f"/images/{index}.jpg" for index in range(10)]}
        first = split_dataset(data, seed=11)
        second = split_dataset(data, seed=11)
        self.assertEqual(first, second)

    def test_class_requires_two_distinct_source_images(self):
        data = {
            "class": [
                "/images/11111111-1111-4111-8111-111111111111___leaf.jpg",
                "/images/11111111-1111-4111-8111-111111111111___leaf_90deg.jpg",
            ]
        }
        with self.assertRaisesRegex(TrainingError, "distinct source images"):
            split_dataset(data)


class ImagePipelineTests(unittest.TestCase):
    def test_preprocessing_returns_rgb_model_tensor(self):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        result = dip_pipeline.preprocess_for_model(image, 224)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_uniform_image_does_not_get_claimed_as_a_leaf(self):
        image = np.full((100, 100, 3), 120, dtype=np.uint8)
        mask, found = dip_pipeline.segment_leaf(image)
        self.assertFalse(found)
        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_moderately_dark_green_patch_is_not_overmarked(self):
        hsv = np.full((200, 200, 3), (60, 180, 200), dtype=np.uint8)
        hsv[80:120, 80:120, 2] = 150
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        leaf_mask = np.zeros((200, 200), dtype=np.uint8)
        leaf_mask[20:180, 20:180] = 255

        lesion_mask, _, lesion_pixels = dip_pipeline.find_lesions(image, leaf_mask)

        self.assertEqual(lesion_pixels, 0)
        self.assertEqual(int(np.count_nonzero(lesion_mask)), 0)


if __name__ == "__main__":
    unittest.main()
