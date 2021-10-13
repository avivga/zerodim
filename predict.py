# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

# Before running cog build on this project, you need to download the
# weights from
# https://drive.google.com/drive/folders/1RXqLyo43uhRx5laTlEHQtnpZssotPPQK
# into the project directory

import shutil
import sys

sys.path.insert(0, "/stylegan2-pytorch")
sys.path.insert(0, "/stylegan-encoder")
import tempfile
from pathlib import Path
import imageio
import PIL.Image
import numpy as np
import cog
from network.training import Model
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


FACTORS = ["age", "gender", "hair_color", "beard", "glasses"]


class Predictor(cog.Predictor):
    def setup(self):
        """Load model into memory so predictions are fast"""

        self.landmarks_detector = LandmarksDetector(
            "/shape_predictor_68_face_landmarks.dat"
        )
        self.model = Model.load("zerodim-ffhq-x256")

    @cog.input(
        "image",
        type=Path,
        help="input facial image. NOTE: image will be aligned and resized to 256*256",
    )
    @cog.input(
        "factor",
        type=str,
        options=FACTORS,
        default="glasses",
        help="attribute of interest",
    )
    def predict(self, image, factor):
        """Run a single preciction"""

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            input_path = str(image)
            # webcam input might be rgba, convert to rgb first
            input = imageio.imread(input_path)
            if input.shape[-1] == 4:
                rgba_image = PIL.Image.open(input_path)
                rgb_image = rgba_image.convert("RGB")
                input_path = str(tmp_dir / "rgb_input.png")
                imageio.imwrite(input_path, rgb_image)

            out_path = Path(tempfile.mkdtemp()) / "out.png"
            aligned_path = str(tmp_dir / "aligned.png")

            self.align_image(input_path, aligned_path)
            if not Path(aligned_path).exists():
                raise Exception("Failed to detect face in image")

            img = PIL.Image.open(aligned_path)
            img = np.array(img.resize((256, 256)))
            manipulated_imgs = self.model.manipulate(
                img, factor, concat_axis=0, include_original=False
            )

            imageio.imwrite(out_path, manipulated_imgs)

            return out_path
        finally:
            # image and out_path is automatically cleaned up by Cog,
            # but we need to clean up other temp files ourselves
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def align_image(self, raw_img_path, aligned_face_path):
        for face_landmarks in self.landmarks_detector.get_landmarks(raw_img_path):
            image_align(raw_img_path, aligned_face_path, face_landmarks)
