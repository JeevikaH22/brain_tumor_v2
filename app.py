import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon=":brain:",
    layout="wide"
)


st.markdown("""
<style>

/* ---- Seaside Drift palette ----
   #F4F6F7  lightest (used sparingly, for primary text only)
   #D2D7DB  light gray (secondary text)
   #A2ACB3  mid gray (borders / muted text)
   #2AA7A1  teal (accent)
   #0B3C49  deep teal navy (base background)
*/

html, body, [class*="css"]{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.stApp{
    background: #0B3C49;
}

/* Streamlit's top header bar — match app background */
[data-testid="stHeader"]{
    background: #0B3C49;
}

/* Kill default streamlit top padding a bit */
.block-container{
    padding-top: 3.5rem;
    max-width: 1180px;
}

/* Header */
.app-header{
    padding: 4px 0 28px 0;
    border-bottom: 1px solid rgba(162,172,179,0.35);
    margin-bottom: 28px;
    margin-top: 12px;
}
.app-header .eyebrow{
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2AA7A1;
    margin-bottom: 10px;
}
.app-header h1{
    font-size: 2rem;
    font-weight: 700;
    color: #F4F6F7;
    margin: 0;
}
.app-header p{
    color: #D2D7DB;
    font-size: 0.98rem;
    margin-top: 6px;
}

/* Native bordered container styling */
[data-testid="stVerticalBlockBorderWrapper"]{
    background: #12475570;
    background: rgba(42,167,161,0.08);
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
[data-testid="stVerticalBlockBorderWrapper"] > div{
    border-color: rgba(162,172,179,0.35) !important;
}

.panel-title{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #D2D7DB;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title .dot{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2AA7A1;
    display: inline-block;
}

/* Make default Streamlit text (captions, labels, uploaded filename) visible on dark bg */
p, span, label, .stMarkdown, [data-testid="stFileUploaderFileName"], [data-testid="stCaptionContainer"]{
    color: #D2D7DB;
}

/* Result banner */
.result-banner{
    border-radius: 10px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.05rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.result-clear{
    background: rgba(42,167,161,0.18);
    border-color: #2AA7A1;
    color: #F4F6F7;
}
.result-tumor{
    background: rgba(220,80,70,0.18);
    border-color: #d65f52;
    color: #F4F6F7;
}
.result-badge{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.badge-clear{ background: #2AA7A1; }
.badge-tumor{ background: #e2685a; }

/* Empty state */
.empty-state{
    text-align: center;
    color: #A2ACB3;
    padding: 70px 20px;
    font-size: 0.95rem;
}
.empty-state .icon-box{
    width: 48px;
    height: 48px;
    margin: 0 auto 14px auto;
    border-radius: 10px;
    background: rgba(162,172,179,0.15);
    border: 1px solid rgba(162,172,179,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #D2D7DB;
    font-size: 1.3rem;
}

/* Buttons */
.stButton>button{
    background: #2AA7A1;
    color: #0B3C49;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover{
    background: #33bdb6;
}

/* File uploader */
[data-testid="stFileUploader"]{
    background: rgba(162,172,179,0.10);
    border-radius: 10px;
    padding: 8px;
    border: 1px dashed #A2ACB3;
}
[data-testid="stFileUploaderDropzone"]{
    background: transparent;
}
[data-testid="stBaseButton-secondary"]{
    background: #A2ACB3;
    color: #0B3C49;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: #082E38;
}
section[data-testid="stSidebar"] *{
    color: #F4F6F7 !important;
}
section[data-testid="stSidebar"] hr{
    border-color: rgba(162,172,179,0.35) !important;
}

.disclaimer{
    font-size: 0.78rem;
    color: #A2ACB3 !important;
    line-height: 1.5;
}

.class-chip{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.9rem;
}
.chip-dot{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="app-header">
    <div class="eyebrow">Diagnostic Support Tool</div>
    <h1>Brain Tumor Classification</h1>
    <p>Upload a brain MRI scan to classify tumor type using a CNN model.</p>
</div>
""", unsafe_allow_html=True)

class_labels = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

class_colors = {
    "Glioma": "#ef4444",
    "Meningioma": "#f59e0b",
    "No Tumor": "#22c55e",
    "Pituitary": "#8b5cf6",
}

transform_pipeline = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = BrainTumorCNN(len(class_labels))
    model.load_state_dict(
        torch.load(r"E:\coding\ML\brain_tumor_project\models\brain_tumor_cnn.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model


model = load_model()

left_col, right_col = st.columns(2, gap="large")

with left_col:
    panel = st.container(border=True)
    with panel:
        st.markdown(
            '<div class="panel-title"><span class="dot"></span>MRI Scan</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose an MRI image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon-box">MRI</div>
                No image uploaded yet<br>Drag and drop a file, or click browse above
            </div>
            """, unsafe_allow_html=True)

with right_col:
    panel = st.container(border=True)
    with panel:
        st.markdown(
            '<div class="panel-title"><span class="dot"></span>Prediction</div>',
            unsafe_allow_html=True
        )

        if uploaded_file is not None:
            tensor = transform_pipeline(image).unsqueeze(0).to(device)

            with st.spinner("Analyzing MRI..."):
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    _, predicted = torch.max(probabilities, 0)
                    prediction = class_labels[predicted.item()]

            if prediction == "No Tumor":
                st.markdown(
                    '<div class="result-banner result-clear">'
                    '<span class="result-badge badge-clear"></span>No Tumor Detected</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-banner result-tumor">'
                    f'<span class="result-badge badge-tumor"></span>Tumor Detected: {prediction}</div>',
                    unsafe_allow_html=True
                )

            st.write("")
            st.caption("For educational/demo purposes only — not a medical diagnosis.")

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon-box">AI</div>
                Prediction will appear here once you upload an MRI image
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.markdown("""
**Model:** Custom CNN
**Framework:** PyTorch
**Input size:** 150 × 150
""")

    st.markdown("---")
    st.subheader("Classes")
    for label in class_labels:
        color = class_colors.get(label, "#2563eb")
        st.markdown(
            f'<div class="class-chip"><span class="chip-dot" style="background:{color};"></span>{label}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<p class="disclaimer">This app is a demo and should not be used for actual '
        'medical diagnosis. Always consult a qualified radiologist or physician.</p>',
        unsafe_allow_html=True
    )