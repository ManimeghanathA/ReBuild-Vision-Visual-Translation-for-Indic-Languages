import os
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random


class FontDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--style_channel', type=int, default=6)
        parser.set_defaults(load_size=64, num_threads=4)

        if is_train:
            parser.set_defaults(
                display_freq=51200,
                update_html_freq=51200,
                print_freq=51200,
                save_latest_freq=5000000,
                n_epochs=10,
                n_epochs_decay=10
            )
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # 🔥 FIXED LANGUAGES
        self.content_language = "tamil"
        self.style_language = "telugu"

        self.style_channel = opt.style_channel

        # Root: datasets/font/train/tamil
        self.root = os.path.join(opt.dataroot, opt.phase)

        self.tamil_dir = os.path.join(self.root, "tamil")
        self.telugu_dir = os.path.join(self.root, "telugu")
        self.source_dir = os.path.join(self.root, "source")

        # Get all Tamil target images
        self.paths = sorted(make_dataset(self.tamil_dir, opt.max_dataset_size))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):

        gt_path = self.paths[index]

        # Extract font + filename
        # .../tamil/<font>/<char>.png
        font_name = os.path.basename(os.path.dirname(gt_path))
        char_name = os.path.basename(gt_path)

        # -------------------------
        # Content (Tamil source)
        # -------------------------
        content_path = os.path.join(self.source_dir, char_name)

        # -------------------------
        # Style (Telugu same font)
        # -------------------------
        style_font_dir = os.path.join(self.telugu_dir, font_name)

        style_files = os.listdir(style_font_dir)

        selected_styles = random.sample(
            style_files,
            min(self.style_channel, len(style_files))
        )

        style_paths = [
            os.path.join(style_font_dir, f)
            for f in selected_styles
        ]

        # -------------------------
        # Load images
        # -------------------------
        content_image = self.load_image(content_path)
        gt_image = self.load_image(gt_path)

        style_images = torch.cat([
            self.load_image(p) for p in style_paths
        ], dim=0)

        return {
            'gt_images': gt_image,
            'content_images': content_image,
            'style_images': style_images,
            'style_image_paths': style_paths,
            'image_paths': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def load_image(self, path):
        img = Image.open(path).convert("L")  # grayscale
        return self.transform(img)