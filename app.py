
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pydicom
from torchvision.models import resnet18


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SarcopeniaAI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "sarcopenia_multimodal_final.pth"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "clinical_scaler.pkl"
)

GRADCAM_PATH = (
    BASE_DIR
    / "assets"
    / "multimuscle_gradcam.png"
)


CLINICAL_FEATURES = [
    "Age",
    "Taille",
    "Poids",
    "Sexe"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MultimodalFusionNetwork(nn.Module):

    def __init__(
        self,
        clinical_input_dim=4
    ):

        super().__init__()

        backbone = resnet18(
            weights=None
        )

        self.image_encoder = nn.Sequential(
            *list(
                backbone.children()
            )[:-1]
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(
                clinical_input_dim,
                16
            ),
            nn.ReLU(),
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                16,
                8
            ),
            nn.ReLU()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(
                512 + 8,
                128
            ),
            nn.ReLU(),
            nn.Dropout(
                0.40
            ),
            nn.Linear(
                128,
                32
            ),
            nn.ReLU(),
            nn.Dropout(
                0.20
            ),
            nn.Linear(
                32,
                1
            )
        )


    def encode_image(
        self,
        image
    ):

        features = self.image_encoder(
            image
        )

        return torch.flatten(
            features,
            1
        )


    def forward(
        self,
        rf,
        vl,
        ta,
        clinical
    ):

        rf_features = self.encode_image(
            rf
        )

        vl_features = self.encode_image(
            vl
        )

        ta_features = self.encode_image(
            ta
        )

        muscle_features = torch.stack(
            [
                rf_features,
                vl_features,
                ta_features
            ],
            dim=1
        ).mean(
            dim=1
        )

        clinical_features = (
            self.clinical_encoder(
                clinical
            )
        )

        fused_features = torch.cat(
            [
                muscle_features,
                clinical_features
            ],
            dim=1
        )

        return self.fusion_classifier(
            fused_features
        )


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

@st.cache_resource
def load_resources():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    model = MultimodalFusionNetwork(
        clinical_input_dim=4
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMAGENET_MEAN = torch.tensor(
    [
        0.485,
        0.456,
        0.406
    ]
).view(
    3,
    1,
    1
)

IMAGENET_STD = torch.tensor(
    [
        0.229,
        0.224,
        0.225
    ]
).view(
    3,
    1,
    1
)


def prepare_rgb_array(
    image_array
):

    image_array = np.asarray(
        image_array
    )


    # --------------------------------------------------------
    # GRAYSCALE → RGB
    # --------------------------------------------------------

    if image_array.ndim == 2:

        image_array = np.stack(
            [
                image_array,
                image_array,
                image_array
            ],
            axis=-1
        )


    # --------------------------------------------------------
    # CHANNEL FIRST → CHANNEL LAST
    # --------------------------------------------------------

    if (
        image_array.ndim == 3
        and image_array.shape[0] in [1, 3]
        and image_array.shape[-1] not in [1, 3, 4]
    ):

        image_array = np.moveaxis(
            image_array,
            0,
            -1
        )


    # --------------------------------------------------------
    # REMOVE ALPHA
    # --------------------------------------------------------

    if (
        image_array.ndim == 3
        and image_array.shape[-1] == 4
    ):

        image_array = image_array[
            ...,
            :3
        ]


    # --------------------------------------------------------
    # SINGLE CHANNEL → RGB
    # --------------------------------------------------------

    if (
        image_array.ndim == 3
        and image_array.shape[-1] == 1
    ):

        image_array = np.repeat(
            image_array,
            3,
            axis=-1
        )


    # --------------------------------------------------------
    # NORMALIZE NON-8-BIT DATA FOR DISPLAY/INFERENCE
    # --------------------------------------------------------

    if image_array.dtype != np.uint8:

        image_array = image_array.astype(
            np.float32
        )

        minimum = np.nanmin(
            image_array
        )

        maximum = np.nanmax(
            image_array
        )

        if maximum > minimum:

            image_array = (
                image_array
                - minimum
            ) / (
                maximum
                - minimum
            )

            image_array = (
                image_array
                * 255.0
            )

        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(
            np.uint8
        )


    if (
        image_array.ndim != 3
        or image_array.shape[-1] != 3
    ):

        raise ValueError(
            "Could not convert uploaded image to RGB."
        )


    return image_array


def read_uploaded_image(
    uploaded_file
):

    file_name = (
        uploaded_file.name.lower()
    )

    raw_bytes = (
        uploaded_file.getvalue()
    )


    # --------------------------------------------------------
    # DICOM
    # --------------------------------------------------------

    if (
        file_name.endswith(".dcm")
        or "." not in uploaded_file.name
    ):

        dicom = pydicom.dcmread(
            io.BytesIO(
                raw_bytes
            )
        )

        image_array = (
            dicom.pixel_array
        )


    # --------------------------------------------------------
    # PNG / JPG / JPEG
    # --------------------------------------------------------

    else:

        image = Image.open(
            io.BytesIO(
                raw_bytes
            )
        ).convert(
            "RGB"
        )

        image_array = np.array(
            image
        )


    return prepare_rgb_array(
        image_array
    )


def preprocess_image(
    image_array
):

    tensor = torch.from_numpy(
        image_array.copy()
    )

    tensor = tensor.permute(
        2,
        0,
        1
    ).float() / 255.0

    tensor = tensor.unsqueeze(
        0
    )

    tensor = F.interpolate(
        tensor,
        size=(
            224,
            224
        ),
        mode="bilinear",
        align_corners=False
    )

    tensor = tensor.squeeze(
        0
    )

    tensor = (
        tensor
        - IMAGENET_MEAN
    ) / IMAGENET_STD

    return tensor


# ============================================================
# INFERENCE
# ============================================================

def predict(
    model,
    scaler,
    rf_array,
    vl_array,
    ta_array,
    age,
    height,
    weight,
    sex
):

    rf = preprocess_image(
        rf_array
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    vl = preprocess_image(
        vl_array
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    ta = preprocess_image(
        ta_array
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )


    clinical_df = pd.DataFrame(
        [
            {
                "Age":
                    float(age),

                "Taille":
                    float(height),

                "Poids":
                    float(weight),

                "Sexe":
                    float(sex)
            }
        ],
        columns=CLINICAL_FEATURES
    )


    clinical_scaled = (
        scaler.transform(
            clinical_df
        )
    )


    clinical_tensor = torch.tensor(
        clinical_scaled,
        dtype=torch.float32
    ).to(
        DEVICE
    )


    with torch.no_grad():

        logits = model(
            rf,
            vl,
            ta,
            clinical_tensor
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .squeeze()
            .item()
        )


    prediction = int(
        probability >= 0.5
    )

    return (
        probability,
        prediction
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "SarcopeniaAI"
)

st.subheader(
    "Explainable Multimodal AI for Muscle Health Research"
)

st.write(
    """
    SarcopeniaAI combines ultrasound images from three muscle
    groups with basic clinical information to demonstrate a
    multimodal deep-learning pipeline for low-muscle-strength
    research.
    """
)


st.warning(
    "Research and educational prototype only. "
    "This application is NOT a medical diagnostic tool."
)


# ============================================================
# ABOUT PIPELINE
# ============================================================

with st.expander(
    "How does the model work?"
):

    st.markdown(
        """
        **Inputs**

        1. Rectus Femoris (RF) ultrasound
        2. Vastus Lateralis (VL) ultrasound
        3. Tibialis Anterior (TA) ultrasound
        4. Age
        5. Height
        6. Weight
        7. Sex

        **Model**

        - ResNet18 extracts ultrasound representations.
        - A clinical MLP processes clinical variables.
        - RF, VL and TA information is combined.
        - Image and clinical representations are fused.
        - The network outputs a research probability for
          low muscle strength.

        **Important:** the development dataset is small,
        so predictions should not be interpreted clinically.
        """
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    model, scaler = load_resources()

except Exception as error:

    st.error(
        f"Could not load model resources: {error}"
    )

    st.stop()


# ============================================================
# INPUT AREA
# ============================================================

st.header(
    "1. Upload Muscle Ultrasound Images"
)

st.caption(
    "Supported formats: DICOM (.dcm), PNG, JPG and JPEG."
)


col1, col2, col3 = st.columns(
    3
)


with col1:

    rf_file = st.file_uploader(
        "Rectus Femoris (RF)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        key="rf"
    )


with col2:

    vl_file = st.file_uploader(
        "Vastus Lateralis (VL)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        key="vl"
    )


with col3:

    ta_file = st.file_uploader(
        "Tibialis Anterior (TA)",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg"
        ],
        key="ta"
    )


# ============================================================
# PREVIEW
# ============================================================

rf_array = None
vl_array = None
ta_array = None


if rf_file is not None:

    try:

        rf_array = read_uploaded_image(
            rf_file
        )

        with col1:

            st.image(
                rf_array,
                caption="RF Preview",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"RF image error: {error}"
        )


if vl_file is not None:

    try:

        vl_array = read_uploaded_image(
            vl_file
        )

        with col2:

            st.image(
                vl_array,
                caption="VL Preview",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"VL image error: {error}"
        )


if ta_file is not None:

    try:

        ta_array = read_uploaded_image(
            ta_file
        )

        with col3:

            st.image(
                ta_array,
                caption="TA Preview",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"TA image error: {error}"
        )


# ============================================================
# CLINICAL INPUT
# ============================================================

st.header(
    "2. Enter Clinical Information"
)


clinical_col1, clinical_col2 = (
    st.columns(
        2
    )
)


with clinical_col1:

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=65,
        step=1
    )

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=220.0,
        value=165.0,
        step=0.1
    )


with clinical_col2:

    weight = st.number_input(
        "Weight (kg)",
        min_value=25.0,
        max_value=250.0,
        value=65.0,
        step=0.1
    )

    sex_label = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )


# Dataset coding used during training:
# 0 = female
# 1 = male

sex = (
    0
    if sex_label == "Female"
    else 1
)


# ============================================================
# PREDICTION
# ============================================================

st.header(
    "3. Run Multimodal Assessment"
)


ready = all(
    image is not None
    for image in [
        rf_array,
        vl_array,
        ta_array
    ]
)


if not ready:

    st.info(
        "Upload RF, VL and TA ultrasound images "
        "to enable assessment."
    )


if st.button(
    "Analyze Patient",
    type="primary",
    disabled=not ready,
    use_container_width=True
):

    try:

        with st.spinner(
            "Analyzing multimodal patient data..."
        ):

            probability, prediction = predict(
                model=model,
                scaler=scaler,
                rf_array=rf_array,
                vl_array=vl_array,
                ta_array=ta_array,
                age=age,
                height=height,
                weight=weight,
                sex=sex
            )


        st.header(
            "Assessment Result"
        )


        result_col1, result_col2 = (
            st.columns(
                2
            )
        )


        with result_col1:

            st.metric(
                "Model Probability",
                f"{probability:.1%}"
            )


        with result_col2:

            if prediction == 1:

                st.metric(
                    "Model Output",
                    "Higher Probability"
                )

            else:

                st.metric(
                    "Model Output",
                    "Lower Probability"
                )


        st.progress(
            float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )
        )


        if prediction == 1:

            st.warning(
                "The research model produced a probability "
                "above its 0.50 classification threshold."
            )

        else:

            st.success(
                "The research model produced a probability "
                "below its 0.50 classification threshold."
            )


        st.caption(
            "This output is a machine-learning research "
            "prediction, not a clinical diagnosis."
        )


        with st.expander(
            "Technical details"
        ):

            st.write(
                f"Device: {DEVICE}"
            )

            st.write(
                f"Threshold: 0.50"
            )

            st.write(
                "Image encoder: ResNet18"
            )

            st.write(
                "Muscles: RF + VL + TA"
            )

            st.write(
                "Clinical variables: "
                "Age + Height + Weight + Sex"
            )


    except Exception as error:

        st.error(
            f"Inference failed: {error}"
        )


# ============================================================
# EXPLAINABILITY EXAMPLE
# ============================================================

st.header(
    "Explainability"
)

st.write(
    """
    Grad-CAM can be used to inspect which ultrasound regions
    influence the image branch of the network. The example
    below was generated from the research experiment.
    """
)


if GRADCAM_PATH.exists():

    st.image(
        str(
            GRADCAM_PATH
        ),
        caption=(
            "Example multi-muscle Grad-CAM visualization"
        ),
        use_container_width=True
    )


st.caption(
    "Grad-CAM visualizations indicate model attention patterns; "
    "they do not establish clinical causality."
)


# ============================================================
# LIMITATIONS
# ============================================================

st.header(
    "Limitations"
)

st.markdown(
    """
    - The development dataset contains a small number of patients.
    - Only patients with complete RF, VL and TA ultrasound data
      were available to the multimodal experiment.
    - The held-out multimodal test subset contained only seven
      patients.
    - The experimental model showed limited generalization.
    - External clinical validation has not been performed.
    - The application must not be used for treatment or diagnosis.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SarcopeniaAI — PyTorch multimodal AI research prototype"
)
