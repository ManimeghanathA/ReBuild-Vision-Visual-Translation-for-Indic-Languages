from PIL import ImageColor
from matplotlib import colors
from PIL import ImageColor
from email.mime import text
from pathlib import Path
import json
import cv2
import torch
import numpy as np
import regex
from PIL import Image, ImageDraw, ImageFont
import textwrap
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from collections import Counter


class Renderer:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.netG = torch.load(
            self.project_root / "generator1.pt",
            map_location=self.device,
            weights_only=False
        )
        self.netG.eval()
        self.netG.to(self.device)
        torch.set_grad_enabled(False)

    def _to_tensor(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).float().unsqueeze(0)
    
    def _tensor_to_image(self, tensor):
        img = tensor.squeeze().cpu().numpy()
        img = (img * 0.5 + 0.5) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def render(self, image_name: str):

        project_root = Path(__file__).resolve().parents[2]

        # -----------------------------
        # Input paths
        # -----------------------------

        original_image_path = (
            project_root /
            "data" /
            "images" /
            f"{image_name}.jpg"
        )

        output_dir = (
            project_root /
            "output" /
            image_name
        )

        inpainted_path = output_dir / "inpainted.jpg"
        metadata_path = output_dir / "metadata.json"
        text_results_path = output_dir / "text_results.json"

        # -----------------------------
        # Output directory
        # -----------------------------

        render_dir = (
            project_root /
            "render_output" /
            image_name
        )

        render_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Load files
        # -----------------------------

        original = cv2.imread(str(original_image_path))
        inpainted = cv2.imread(str(inpainted_path))

        if original is None:
            raise FileNotFoundError(original_image_path)

        if inpainted is None:
            raise FileNotFoundError(inpainted_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(text_results_path, "r", encoding="utf-8") as f:
            text_results = json.load(f)

        print(f"Loaded image: {image_name}")
        print(f"Detected areas : {len(metadata['areas'])}")
        print(f"Translations   : {len(text_results)}")

        # Temporary
        '''
        return {
            "original": original,
            "inpainted": inpainted,
            "metadata": metadata,
            "text_results": text_results,
            "render_dir": render_dir,
        }
        '''
        #self._save_warped_quads(original,metadata,render_dir)
        self._extract_style_images(
            original,
            metadata,
            render_dir
        )

        print("Style images extracted.")

        for area, result in zip(metadata["areas"], text_results):

            area_index = area["area_index"]

            layout = self._layout_text(
                result["tamil_translation"],
                result["area_bbox"]
            )

            style_imgs = []

            for i in range(1, 4):

                style = cv2.imread(
                    str(
                        render_dir /
                        f"area{area_index}_style{i}.png"
                    ),
                    cv2.IMREAD_GRAYSCALE
                )

                style_imgs.append(style)
            #FAKE
            style_imgs1 = []

            for i in range(1, 4):

                style1 = cv2.imread(
                    str(
                        render_dir /
                        f"area{area_index}_style{i}.png"
                    )
                )

                style_imgs1.append(style1)
            text_color = self._estimate_style_color_2(style_imgs1)
            print(text_color)

            # --------------------------------------------------
            # Temporary generator sanity check
            # --------------------------------------------------

            glyph = self._generate_glyph(
                "ம",
                style_imgs
            )

            cv2.imwrite(
                str(
                    render_dir /
                    f"area{area_index}_test_glyph.png"
                ),
                glyph
            )

            # --------------------------------------------------
            # Compose paragraph
            # --------------------------------------------------

            paragraph = self._compose_paragraph(
                layout["lines"],
                style_imgs
            )

            #test
            _, bw = cv2.threshold(paragraph, 180, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2))

            dilated = cv2.dilate(bw, kernel, iterations=1)

            paragraph = 255 - dilated

            cv2.imwrite(
                str(
                    render_dir /
                    f"area{area_index}_paragraph.png"
                ),
                paragraph
            )
            

            paragraph = self._compose_paragraph(layout["lines"],style_imgs)

            #test
            '''
            _, bw = cv2.threshold(paragraph, 180, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2))

            dilated = cv2.dilate(bw, kernel, iterations=1)

            paragraph = 255 - dilated
            '''

            paragraph = self._colorize_paragraph(paragraph,text_color)
            
            '''
            cv2.imwrite(
                str(render_dir / f"area{area_index}_colored.png"),
                paragraph
            )
            '''
            

            paragraph = self._fit_to_bbox(paragraph,result["area_bbox"])

            '''
            cv2.imwrite(
                str(render_dir / f"area{area_index}_fit.png"),
                paragraph
            )
            '''

            inpainted = self._paste_paragraph_2(inpainted,paragraph,result["area_bbox"])

        cv2.imwrite(str(render_dir / "rendered.jpg"),inpainted)

        
        '''
        #print("Warped quads saved.")

        for area, result in zip(metadata["areas"], text_results):

            layout = self._layout_text(
                result["tamil_translation"],
                result["area_bbox"]
            )

            print()

            print(layout["font_size"])

            for line in layout["lines"]:
                print(line)

        style_imgs = []

        for i in range(1, 4):

            style = cv2.imread(
                str(
                    render_dir /
                    f"area0_style{i}.png"
                )
            )

            style_imgs.append(style)

        glyph = self._generate_glyph(
            "ம",
            style_imgs
        )

        cv2.imwrite(
            str(render_dir / "test_glyph.png"),
            glyph
        )

        layout = self._layout_text(
            result["tamil_translation"],
            result["area_bbox"]
        )

        style_imgs = []

        for i in range(1, 4):

            style = cv2.imread(
                str(
                    render_dir /
                    f"area{area_index}_style{i}.png"
                )
            )

            style_imgs.append(style)

        paragraph = self._compose_paragraph(
            layout["lines"],
            style_imgs
        )

        cv2.imwrite(
            str(
                render_dir /
                f"area{area_index}_paragraph.png"
            ),
            paragraph
        )

        '''

        






    def _warp_quad(self, image, quad):
        """
        Perspective-warp a quadrilateral region into a straight rectangle.

        Parameters
        ----------
        image : np.ndarray
            Original BGR image.

        quad : list
            Four corner points from metadata.json in the order:
            [top-left, top-right, bottom-right, bottom-left]

        Returns
        -------
        np.ndarray
            Warped BGR image.
        """

        pts = np.array(quad, dtype=np.float32)

        # Compute output size
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])
        height = int(max(height_left, height_right))

        width = max(width, 1)
        height = max(height, 1)

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts, dst)

        warped = cv2.warpPerspective(
            image,
            M,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped
    
    def _save_warped_quads(self, original, metadata, render_dir):
        for area in metadata["areas"]:

            area_index = area["area_index"]

            for quad_index, quad in enumerate(area["area_quads"]):

                warped = self._warp_quad(original, quad)

                filename = (
                    render_dir /
                    f"area{area_index}_quad{quad_index}.png"
                )

                #cv2.imwrite(str(filename), warped)
                candidates = self._extract_candidates(warped)
                for i, candidate in enumerate(candidates):

                    cv2.imwrite(
                        str(
                            render_dir /
                            f"area{area_index}_quad{quad_index}_cand{i}.png"
                        ),
                        255 - candidate
                    )
    
    def _extract_candidates(self, warped):
        """
        Extract character candidates from a warped Telugu word image.

        Parameters
        ----------
        warped : np.ndarray
            Perspective-corrected BGR image.

        Returns
        -------
        list[np.ndarray]
            Cropped grayscale candidate glyphs.
        """

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Black text on white background
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h, w = binary.shape

        # --------------------------------------------------
        # Vertical projection
        # --------------------------------------------------

        projection = np.sum(binary > 0, axis=0)

        # Smooth the projection slightly
        projection = np.convolve(
            projection,
            np.ones(5) / 5,
            mode="same"
        )

        # Columns with almost no ink
        threshold = 0.05 * h
        whitespace = projection < threshold

        # --------------------------------------------------
        # Find cut positions
        # --------------------------------------------------

        segments = []

        start = None

        min_gap = 4

        x = 0

        while x < w:

            if not whitespace[x]:

                if start is None:
                    start = x

            else:

                gap_start = x

                while x < w and whitespace[x]:
                    x += 1

                gap_width = x - gap_start

                if gap_width >= min_gap and start is not None:

                    end = gap_start

                    if end - start > 8:
                        segments.append((start, end))

                    start = None

                continue

            x += 1

        if start is not None:
            segments.append((start, w))

        # --------------------------------------------------
        # Crop candidates
        # --------------------------------------------------

        candidates = []

        for left, right in segments:

            crop = binary[:, left:right]

            ys, xs = np.where(crop > 0)

            if len(xs) == 0:
                continue

            x0 = xs.min()
            x1 = xs.max()

            y0 = ys.min()
            y1 = ys.max()

            glyph = crop[y0:y1+1, x0:x1+1]

            if glyph.shape[0] < 15 or glyph.shape[1] < 15:
                continue

            candidates.append(glyph)

        return candidates
    
    def _pad_to_square(self, image):
        """
        Pad an image to a square using a white background.
        """

        if len(image.shape) == 2:
            h, w = image.shape
            channels = None
        else:
            h, w, channels = image.shape

        size = max(h, w)

        if channels is None:
            square = np.full((size, size), 255, dtype=np.uint8)
        else:
            square = np.full((size, size, channels), 255, dtype=np.uint8)

        y = (size - h) // 2
        x = (size - w) // 2

        square[y:y+h, x:x+w] = image

        return square

    def _score_quad(self, warped):
        """
        Score a warped quad for style quality.

        Larger, ink-rich quads receive higher scores.
        """

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h, w = binary.shape

        area = h * w

        ink_density = np.count_nonzero(binary) / area

        score = (
            0.6 * area +
            0.4 * ink_density * area
        )

        return score

    def _extract_style_images(self, original, metadata, render_dir):

        for area in metadata["areas"]:

            area_index = area["area_index"]

            candidates = []

            # Pair every quad with its OCR text
            for quad, ocr in zip(area["area_quads"], area["raw_ocr"]):

                warped = self._warp_quad(original, quad)

                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                _, binary = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

                h, w = binary.shape

                if h < 20 or w < 20:
                    continue

                grapheme_count = len(regex.findall(r"\X", ocr["text"]))

                ink_density = np.count_nonzero(binary) / (h * w)

                aspect_ratio = max(w, h) / min(w, h)

                candidates.append({
                    "image": warped,
                    #"color": self._extract_text_color(warped),
                    "text": ocr["text"],
                    "graphemes": grapheme_count,
                    "aspect": aspect_ratio,
                    "ink": ink_density
                })

            if len(candidates) == 0:
                continue

            # Prefer:
            # 1. fewer graphemes
            # 2. squarer crops
            # 3. denser text
            candidates.sort(
                key=lambda x: (
                    x["graphemes"],
                    abs(x["aspect"] - 1.0),
                    -x["ink"]
                )
            )

            styles = [c["image"] for c in candidates[:3]]
            '''

            colors = np.array(
                [c["color"] for c in styles],
                dtype=np.float32
            )

            text_color = tuple(
                np.median(colors, axis=0).astype(np.uint8)
            )

            style_imgs = [c["image"] for c in styles]
            '''

            # Duplicate the last available style if needed
            while len(styles) < 3:
                styles.append(styles[-1])

            for i, img in enumerate(styles):

                square = self._pad_to_square(img)

                square = cv2.resize(
                    square,
                    (64, 64),
                    interpolation=cv2.INTER_AREA
                )

                cv2.imwrite(
                    str(
                        render_dir /
                        f"area{area_index}_style{i+1}.png"
                    ),
                    square
                )
        #return style_imgs, text_color
        
        
    def _layout_text(self, text, bbox):
        """
        Determine the largest font size and wrapping that fits inside
        the bounding box.

        Returns
        -------
        dict
            {
                "font_size": int,
                "lines": list[str],
                "line_height": int
            }
        """

        x1, y1, x2, y2 = bbox

        max_width = x2 - x1
        max_height = y2 - y1

        dummy = Image.new("L", (1, 1))
        draw = ImageDraw.Draw(dummy)

        # Start from the maximum possible size
        for font_size in range(max_height, 8, -2):

            font = ImageFont.truetype(
                "NotoSansTamil-Regular.ttf",
                font_size
            )

            words = text.split()

            if len(words) == 0:
                words = [text]

            lines = []
            current = ""

            for word in words:

                candidate = word if current == "" else current + " " + word

                bbox_text = draw.textbbox(
                    (0, 0),
                    candidate,
                    font=font
                )

                width = bbox_text[2] - bbox_text[0]

                if width <= max_width:

                    current = candidate

                else:

                    if current:
                        lines.append(current)

                    current = word

            if current:
                lines.append(current)

            ascent, descent = font.getmetrics()

            line_height = ascent + descent + 4

            total_height = line_height * len(lines)

            if total_height <= max_height:

                return {
                    "font_size": font_size,
                    "lines": lines,
                    "line_height": line_height
                }

        return {
            "font_size": 10,
            "lines": [text],
            "line_height": 12
        }

    def _render_tamil_grapheme(self, grapheme):
        """
        Render a Tamil grapheme into a 64x64 grayscale image.

        This reproduces the content image generation used during
        training of the generator.
        """

        canvas_size = 64
        font_size = 48

        img = Image.new(
            "L",
            (canvas_size, canvas_size),
            color=255
        )

        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(
            str(self.project_root / "NotoSansTamil-Regular.ttf"),
            font_size
        )

        bbox = draw.textbbox(
            (0, 0),
            grapheme,
            font=font
        )

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (canvas_size - text_w) // 2 - bbox[0]
        y = (canvas_size - text_h) // 2 - bbox[1]

        draw.text(
            (x, y),
            grapheme,
            fill=0,
            font=font
        )

        return np.array(img)

    def _generate_glyph(self, grapheme, style_imgs):
        """
        Generate a stylized Tamil grapheme using the generator.
        """

        content_img = self._render_tamil_grapheme(grapheme)

        style_tensors = []

        for img in style_imgs:
            style_tensors.append(self._to_tensor(img))

        style_tensor = torch.cat(
            style_tensors,
            dim=0
        ).unsqueeze(0).to(self.device)

        content_tensor = (
            self._to_tensor(content_img)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            output = self.netG(
                (
                    content_tensor,
                    style_tensor
                )
            )

        return self._tensor_to_image(output)

    def _glyph_bbox(self, glyph):
        """
        Returns the bounding box of the visible ink.

        Parameters
        ----------
        glyph : np.ndarray
            64x64 grayscale image.

        Returns
        -------
        tuple
            (left, top, right, bottom)
        """

        mask = glyph < 200

        ys, xs = np.where(mask)

        if len(xs) == 0:
            return 0, 0, glyph.shape[1], glyph.shape[0]

        return (
            xs.min(),
            ys.min(),
            xs.max() + 1,
            ys.max() + 1
        )
    
    def _compose_line(self, line, style_imgs):
        """
        Generate and compose one rendered Tamil line.

        Parameters
        ----------
        line : str
            One wrapped line of Tamil text.

        style_imgs : list[np.ndarray]
            Three extracted Telugu style images.

        Returns
        -------
        np.ndarray
            Rendered line.
        """

        graphemes = regex.findall(r"\X", line)

        SPACE_WIDTH = 20
        LETTER_SPACING = 4

        glyphs = []

        max_height = 0
        total_width = 0

        # ---------------------------------------
        # Generate every glyph
        # ---------------------------------------

        for grapheme in graphemes:

            if grapheme.isspace():

                glyphs.append({
                    "space": True,
                    "width": SPACE_WIDTH
                })

                total_width += SPACE_WIDTH

                continue

            glyph = self._generate_glyph(
                grapheme,
                style_imgs
            )

            left, top, right, bottom = self._glyph_bbox(glyph)

            cropped = glyph[
                top:bottom,
                left:right
            ]

            h, w = cropped.shape

            glyphs.append({
                "space": False,
                "image": cropped,
                "width": w,
                "height": h
            })

            total_width += w + LETTER_SPACING

            max_height = max(max_height, h)

        if total_width <= 0:
            total_width = 1

        canvas = np.full(
            (
                max_height,
                total_width
            ),
            255,
            dtype=np.uint8
        )

        # ---------------------------------------
        # Paste glyphs
        # ---------------------------------------

        cursor = 0

        for glyph in glyphs:

            if glyph["space"]:

                cursor += SPACE_WIDTH

                continue

            img = glyph["image"]

            h, w = img.shape

            # Bottom alignment
            y = max_height - h

            roi = canvas[
                y:y+h,
                cursor:cursor+w
            ]

            canvas[
                y:y+h,
                cursor:cursor+w
            ] = np.minimum(
                roi,
                img
            )

            cursor += w + LETTER_SPACING

        return canvas

    def _compose_paragraph(self, lines, style_imgs):
        """
        Compose multiple rendered lines into a single paragraph image.

        Parameters
        ----------
        lines : list[str]
            Wrapped Tamil lines.

        style_imgs : list[np.ndarray]
            Three Telugu style images.

        Returns
        -------
        np.ndarray
            Paragraph image.
        """

        LINE_SPACING = 10

        line_images = []

        paragraph_width = 0
        paragraph_height = 0

        # -----------------------------------------
        # Render every line
        # -----------------------------------------

        for line in lines:

            line_img = self._compose_line(
                line,
                style_imgs
            )

            line_images.append(line_img)

            h, w = line_img.shape

            paragraph_width = max(
                paragraph_width,
                w
            )

            paragraph_height += h

        if len(line_images) > 1:
            paragraph_height += (
                LINE_SPACING *
                (len(line_images) - 1)
            )

        paragraph = np.full(
            (
                max(paragraph_height, 1),
                max(paragraph_width, 1)
            ),
            255,
            dtype=np.uint8
        )

        # -----------------------------------------
        # Paste every line centered
        # -----------------------------------------

        y = 0

        for line_img in line_images:

            h, w = line_img.shape

            x = (paragraph_width - w) // 2

            roi = paragraph[
                y:y+h,
                x:x+w
            ]

            paragraph[
                y:y+h,
                x:x+w
            ] = np.minimum(
                roi,
                line_img
            )

            y += h + LINE_SPACING

        return paragraph

    def _fit_to_bbox(self, paragraph, bbox):
        """
        Resize a rendered paragraph so that it fits inside
        the target bounding box while preserving aspect ratio.
        """

        x1, y1, x2, y2 = bbox

        bbox_w = int((x2 - x1) * 0.90)
        bbox_h = int((y2 - y1) * 0.90)

        h, w = paragraph.shape[:2]

        scale = min(
            bbox_w / w,
            bbox_h / h
        )

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(
            paragraph,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        return resized

    def _paste_paragraph(self, image, paragraph, bbox):
        """
        Paste a colored paragraph into the target bbox.
        """

        x1, y1, x2, y2 = bbox

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        h, w, _ = paragraph.shape

        x = x1 + (bbox_w - w) // 2
        y = y1 + (bbox_h - h) // 2

        roi = image[
            y:y+h,
            x:x+w
        ]

        image[
            y:y+h,
            x:x+w
        ] = np.minimum(
            roi,
            paragraph
        )

        return image

    def _extract_text_color(self, warped):
        """
        Estimate the dominant text color from a warped text quad.

        Parameters
        ----------
        warped : np.ndarray
            Warped BGR image.

        Returns
        -------
        tuple
            (B, G, R)
        """

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        mask = binary > 0

        if np.count_nonzero(mask) == 0:
            return (0, 0, 0)

        pixels = warped[mask]

        b = int(np.median(pixels[:, 0]))
        g = int(np.median(pixels[:, 1]))
        r = int(np.median(pixels[:, 2]))

        return (b, g, r)

    def _colorize_paragraph(self, paragraph, color):
        """
        Convert a grayscale paragraph into colored text.

        Parameters
        ----------
        paragraph : np.ndarray
            Grayscale paragraph (0-255).

        color : tuple
            (B,G,R)

        Returns
        -------
        np.ndarray
            Colored BGR paragraph.
        """

        paragraph = paragraph.astype(np.float32)

        alpha = 1.0 - paragraph / 255.0

        h, w = paragraph.shape

        colored = np.full(
            (h, w, 3),
            255,
            dtype=np.float32
        )

        b, g, r = color

        colored[:, :, 0] = 255 * (1 - alpha) + b * alpha
        colored[:, :, 1] = 255 * (1 - alpha) + g * alpha
        colored[:, :, 2] = 255 * (1 - alpha) + r * alpha

        return colored.astype(np.uint8)

    def _estimate_style_color(self, style_imgs):
        """
        Estimate the representative text color from the selected style images.

        Uses image gradients to find text strokes instead of thresholding.
        Works well even with colored backgrounds and padded images.

        Parameters
        ----------
        style_imgs : list[np.ndarray]

        Returns
        -------
        tuple
            (B, G, R)
        """

        colors = []

        for img in style_imgs:

            # Ignore grayscale images if any accidentally slip through
            if img.ndim != 3:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Ignore the white padding
            valid = gray < 245

            if np.count_nonzero(valid) == 0:
                continue

            # Gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

            mag = cv2.magnitude(gx, gy)

            # Keep only strong edges inside the valid region
            edge_mask = (mag > np.percentile(mag[valid], 85)) & valid

            if np.count_nonzero(edge_mask) < 20:
                edge_mask = valid

            pixels = img[edge_mask]

            colors.append([
                np.median(pixels[:, 0]),
                np.median(pixels[:, 1]),
                np.median(pixels[:, 2])
            ])

        if len(colors) == 0:
            return (0, 0, 0)

        colors = np.array(colors)
        color = tuple(np.median(colors, axis=0).astype(np.uint8))
        if min(color) > 190:
            return (0, 0, 0)
        #print(color)
        #print(min(color))
        return min(color)

    #test2

    def _estimate_style_color_2(self, style_imgs):
        """
        Estimate the representative text color using K-Means clustering.

        For each style image:
            - Cluster pixels into 2 colors.
            - Larger cluster -> background
            - Smaller cluster -> foreground/text
        The final color is the majority foreground color across all style images.
        """

        colors = []

        for img in style_imgs:

            if img is None or img.ndim != 3:
                continue

            # Ignore almost-white padding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            valid = gray < 245

            if np.count_nonzero(valid) < 20:
                continue

            pixels = img[valid].reshape((-1, 3)).astype(np.float32)

            # --------------------------
            # KMeans
            # --------------------------

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.2,
            )

            _, labels, centers = cv2.kmeans(
                pixels,
                2,
                None,
                criteria,
                10,
                cv2.KMEANS_PP_CENTERS,
            )

            labels = labels.flatten()

            counts = np.bincount(labels)

            # Smaller cluster = text
            text_cluster = np.argmin(counts)

            text_color = centers[text_cluster]

            colors.append(tuple(text_color.astype(np.uint8)))

        if len(colors) == 0:
            return (0, 0, 0)

        # ---------------------------------------
        # Majority vote with slight quantization
        # ---------------------------------------

        quantized = [
            (
                int(c[0] // 8),
                int(c[1] // 8),
                int(c[2] // 8),
            )
            for c in colors
        ]

        winner = Counter(quantized).most_common(1)[0][0]

        # Average all colors belonging to the winning bucket
        selected = np.array([
            c
            for c, q in zip(colors, quantized)
            if q == winner
        ])

        final = tuple(np.mean(selected, axis=0).astype(np.uint8))

        # Fallback for nearly-white detections
        if min(final) > 190:
            return (0, 0, 0)

        return final

    def _paste_paragraph_2(self, image, paragraph, bbox):
        """
        Paste a colored paragraph into the target bbox using alpha blending.

        Assumes:
            - paragraph has a white background
            - text can be any color
            - anti-aliased edges are preserved
        """

        import cv2
        import numpy as np

        x1, y1, x2, y2 = bbox

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        h, w = paragraph.shape[:2]

        # Center paragraph in bbox
        x = x1 + (bbox_w - w) // 2
        y = y1 + (bbox_h - h) // 2

        # Clip in case paragraph extends outside image
        H, W = image.shape[:2]

        x_start = max(0, x)
        y_start = max(0, y)

        x_end = min(W, x + w)
        y_end = min(H, y + h)

        if x_start >= x_end or y_start >= y_end:
            return image

        roi = image[y_start:y_end, x_start:x_end].astype(np.float32)

        para = paragraph[
            y_start - y:y_end - y,
            x_start - x:x_end - x
        ].astype(np.float32)

        # ---------------------------------------
        # Compute alpha from the white background
        # ---------------------------------------

        gray = cv2.cvtColor(para.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # White -> alpha = 0
        # Black/Colored text -> alpha close to 1
        alpha = 1.0 - gray.astype(np.float32) / 255.0

        # Optional: make text slightly stronger
        alpha = np.clip(alpha * 1.2, 0, 1)

        alpha = alpha[:, :, None]

        # ---------------------------------------
        # Alpha blend
        # ---------------------------------------

        blended = roi * (1.0 - alpha) + para * alpha

        image[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)

        return image