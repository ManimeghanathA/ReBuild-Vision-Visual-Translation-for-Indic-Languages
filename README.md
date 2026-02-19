# ReBuild Vision: Visual Translation for Indic Languages  

ReBuild Vision is an end-to-end deep learning framework designed to detect, translate, remove, and reconstruct scene text embedded within images and videos. Unlike traditional subtitle-based localization systems, this project focuses on **visually embedded text** such as signboards, posters, banners, and environmental captions — enabling immersive cross-lingual adaptation for Indic languages.

The system aims to produce outputs that appear **natively authored in the target language**, rather than simply overlaying translated text on top of the original scene.

---

## 🚀 Problem Statement

Most multimedia localization pipelines focus only on audio dubbing or subtitle generation. However, textual content embedded within visual scenes remains untranslated, breaking immersion and limiting accessibility.

Overlay-based approaches fail to:
- Remove original text cleanly
- Preserve background realism
- Maintain spatial and stylistic consistency
- Adapt translations to geometric constraints

ReBuild Vision addresses these challenges through a fully modular deep learning pipeline that performs **scene-aware visual text reconstruction and translation**.

---

## 🧠 System Architecture

The pipeline is structured into two progressive stages:

### Phase 1: Image-Level Scene Reconstruction (Proof of Concept)

1. **Scene Text Detection**  
   Detects arbitrarily oriented text regions in natural scenes.

2. **Optical Character Recognition (OCR)**  
   Extracts textual content from detected regions.

3. **Text Normalization**  
   A custom Seq2Seq-based normalization model cleans noisy OCR outputs (Telugu normalization dataset created and trained).

4. **Neural Machine Translation**  
   Performs cross-lingual translation (Telugu → Tamil).

5. **Background Inpainting**  
   Removes original text and reconstructs the underlying texture using deep learning models.

6. **Style-Consistent Text Rendering**  
   Re-renders translated text while preserving:
   - Perspective
   - Color distribution
   - Font scale
   - Spatial constraints

---

### Phase 2: Video-Level Extension (Future Ambitions)

- Multi-frame text tracking  
- Motion-aware inpainting  
- Temporal consistency enforcement  
- Flicker reduction  

---

## 🛠 Technologies Used

- Python  
- PyTorch  
- Transformer-based Seq2Seq Models  
- OpenCV  
- CRNN / Transformer-based OCR  
- GAN / Diffusion-based Inpainting   

---

## 📊 Evaluation Metrics

The system is evaluated at multiple stages:

### Text Detection & OCR
- Precision / Recall / F1-score  
- Character Error Rate (CER)  
- Word Error Rate (WER)

### Translation
[Yet to be Done]

### Background Reconstruction
[yet to be Done]

### Video Extension
[yet to be Done]

---

## 🌍 Target Language Pair

Telugu → Tamil  
(Architecture is modular and extensible to additional Indic languages)

---

## 📂 Repository Structure
ReBuild-Vision/
│
├── checkpoints/
├── Dataset/
├── docs/
├── sample images/
├── Telugu Text Normalization- Training code/
└── <Visual_Translation_Area Wise Text Extraction .ipynb>


---

## 🎯 Applications

- Film and media localization  
- Educational content translation  
- Tourism and navigation systems  
- Cross-lingual accessibility tools  
- Regional content adaptation  

---

## 🔮 Future Work

- Multi-language expansion  
- Real-time optimization  
- Joint audio-visual localization  
- Robust handling of dynamic scenes  

---

## 📌 Project Vision

ReBuild Vision aims to move beyond subtitle overlays and toward **true visual localization**, where digital media appears organically created in the target language — improving accessibility, immersion, and cultural adaptation.

---

## 👥 Contributors

- Manimeghanath A   

---
