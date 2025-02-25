import os
import numpy as np
import onnxruntime as rt
import pandas as pd
import huggingface_hub
import torch
from PIL import Image

# Kaomojis list
KAOMOJIS = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

def load_labels(dataframe: pd.DataFrame) -> tuple[list[str], list[int], list[int], list[int]]:
    """Loads and preprocesses tag labels from a Pandas DataFrame."""
    dataframe["name"] = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x)
    tag_names = dataframe["name"].tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs: np.ndarray) -> float:
    """
    Calculates the Maximum Cut Thresholding (MCut) for multi-label classification.
    Ref: Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
         for Multi-label Classification. In 11th International Symposium, IDA 2012
         (pp. 172-183).
    """
    sorted_probs = np.sort(probs)[::-1]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = np.argmax(difs)
    return (sorted_probs[t] + sorted_probs[t + 1]) / 2

class Predictor:
    """Class for managing model loading, image preprocessing, and prediction."""
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = model_cache_dir
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.reset_state()
        self.initialize_session_options()

    def reset_state(self) -> None:
        """Resets the predictor's state variables."""
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None

    def initialize_session_options(self) -> None:
        """Initializes ONNX Runtime session options and CUDA provider options."""
        self.session_options = rt.SessionOptions()
        self.session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }

    def download_model(self, model_repo: str) -> tuple[str, str]:
        """Downloads model and label files from Hugging Face Hub."""
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
            cache_dir=self.model_cache_dir
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
            cache_dir=self.model_cache_dir
        )
        return csv_path, model_path

    def load_model(self, model_repo: str) -> None:
        """Loads the specified model and its associated label file."""
        if model_repo == self.last_loaded_repo and self.model is not None:
            return

        self.reset_state()
        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        self.tag_names, self.rating_indexes, self.general_indexes, self.character_indexes = load_labels(tags_df)

        # Use CUDA provider if available
        providers = [('CUDAExecutionProvider', self.cuda_provider_options)] if torch.cuda.is_available() else []
        self.model = rt.InferenceSession(model_path, providers=providers, sess_options=self.session_options)

        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        """Preprocesses the image for the model."""
        target_size = self.model_target_size

        # Ensure the image is in RGBA mode and create a white canvas
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array and BGR
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # RGB to BGR

        return np.expand_dims(image_array, axis=0)

    def format_tag_with_weight(self, tag: str, weight: float) -> str:
        """Formats a tag with its prediction weight for weighted captions."""
        return f"({tag}:{weight:.2f})"

    def predict(
        self,
        image: Image.Image,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        weighted_captions: bool = False,
    ) -> tuple[str, dict, dict, dict]:
        """
        Predicts tags for a given image using the specified model and thresholds.
        """
        PRIORITY_TAGS = [
            '1girl', '2girls', '3girls', '4girls', '5girls',
            '1boy', '2boys', '3boys', '4boys'
        ]

        self.load_model(model_repo)

        image_array = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_array})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # Process ratings, general tags, and character tags
        rating = dict([labels[i] for i in self.rating_indexes])
        general_names = [labels[i] for i in self.general_indexes]
        character_names = [labels[i] for i in self.character_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = max(0.15, mcut_threshold(character_probs))

        general_res = {x[0]: x[1] for x in general_names if x[1] > general_thresh}
        character_res = {x[0]: x[1] for x in character_names if x[1] > character_thresh}

        # Sort and format tags
        sorted_general = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
        sorted_character = sorted(character_res.items(), key=lambda x: x[1], reverse=True)

        if weighted_captions:
            sorted_general_strings = [self.format_tag_with_weight(x[0], x[1]) for x in sorted_general]
            sorted_character_strings = [self.format_tag_with_weight(x[0], x[1]) for x in sorted_character]
        else:
            sorted_general_strings = [x[0] for x in sorted_general]
            sorted_character_strings = [x[0] for x in sorted_character]

        # Prioritize and concatenate tags
        final_tags = []
        for tag in PRIORITY_TAGS:
            if tag in general_res:
                if weighted_captions:
                    final_tags.append(self.format_tag_with_weight(tag, general_res[tag]))
                else:
                    final_tags.append(tag)
                sorted_general_strings = [s for s in sorted_general_strings
                                          if (s.split(':')[0].strip('()') if weighted_captions else s) != tag]

        if sorted_character_strings:
            final_tags.append(", ".join(sorted_character_strings))

        final_tags.extend(sorted_general_strings)

        sorted_general_strings = ", ".join(final_tags)

        # Escape parentheses if not using weighted captions
        if not weighted_captions:
            sorted_general_strings = sorted_general_strings.replace("(", r"\(").replace(")", r"\)")

        return sorted_general_strings, rating, character_res, general_res

    def batch_predict(
        self,
        image_dir: str,
        model_repo: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
        append_to_existing: bool,
        exclude_character_tags: bool,
        weighted_captions: bool = False,
    ) -> dict[str, str]:
        """
        Predicts tags for a batch of images in a directory.
        """
        results = {}
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path)
                tags, _, character_res, _ = self.predict(
                    image,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                    weighted_captions,
                )

                if exclude_character_tags:
                    # Filter out character tags
                    tags_list = tags.split(', ')
                    if weighted_captions:
                        character_set = {name.split(':')[0].strip('()') for name in character_res.keys()}
                        filtered_tags_list = [tag for tag in tags_list if tag.split(':')[0].strip('()') not in character_set]
                    else:
                        character_set = {name.replace('\\', '') for name in character_res.keys()}
                        filtered_tags_list = [tag for tag in tags_list if tag.replace('\\', '') not in character_set]
                    tags = ', '.join(filtered_tags_list)

                results[filename] = tags

                # Write tags to a text file
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(image_dir, txt_filename)

                if append_to_existing and os.path.exists(txt_path):
                    with open(txt_path, "r+") as f:
                        existing_content = f.read().strip()
                        f.seek(0, 0)
                        if existing_content:
                            f.write(f"{existing_content}, {tags}")
                        else:
                            f.write(tags)
                else:
                    with open(txt_path, "w") as f:
                        f.write(tags)
        return results