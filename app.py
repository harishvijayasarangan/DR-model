import streamlit as st
import onnxruntime
import numpy as np
from PIL import Image
import os

# Set light theme and page layout
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force light theme and set text colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Force light theme */
    html, body {
        color-scheme: light !important;
        background-color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background and text colors */
    .main {background-color: #FFFFFF !important;}
    body {color: #1A4B8C !important;}
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1 !important;
        font-weight: 600 !important;
    }
    
    /* Card styling for content sections */
    .stApp {
        background-color: #F8F9FA;
    }
    
    .css-1kyxreq, .css-1oe6wy4 {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Flag button styling */
    .flag-btn button {
        background-color: #E0E0E0;
        color: #5F6368;
        font-weight: 500;
    }
    .flag-btn button:hover {
        background-color: #BDBDBD;
    }
    
    /* File uploader styling */
    div[data-testid="stFileUploadDropzone"] {
        height: 150px;
        background-color: #F5F7FA;
        border: 2px dashed #BDBDBD;
        border-radius: 10px;
        color: #5F6368;
    }
    
    /* Progress bars styling */
    .confidence-bar {
        height: 12px;
        border-radius: 10px;
        background: linear-gradient(90deg, #1E88E5, #64B5F6);
    }
    
    /* Example buttons styling */
    .example-btn {
        background-color: #F5F7FA;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 0.3rem;
    }
    
    /* Container styling */
    .content-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    /* Result container styling */
    .results-container {
        padding: 15px;
        border-radius: 8px;
        background-color: #F5F7FA;
        margin-top: 10px;
    }
    
    /* Additional Overrides for all text elements */
    
    /* Base text styles */
    * {
        color: #1A4B8C !important;
    }
    
    /* Override Streamlit's default text color for all text */
    .stMarkdown, p, div, span, label, .stMarkdown p, h1, h2, h3, h4, h5, h6,
    .st-emotion-cache-1n76uvr, .st-emotion-cache-183lzff {
        color: #1A4B8C !important;
    }
    
    /* Ensure all headings have explicit color */
    h1, h2, h3, h4, h5, h6, .css-10trblm, .css-681jrp {
        color: #0D47A1 !important;
    }
    
    /* Force text colors for specific elements */
    [data-testid="stVerticalBlock"] *:not(button):not(.stProgress):not(svg):not(path):not(rect):not(polygon):not(circle):not(line):not(ellipse) {
        color: #1A4B8C !important;
    }
    
    /* Info text styling */
    .info-text {
        color: #1565C0;  /* Blue info text */
        font-size: 0.9em;
        font-weight: 500;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #1976D2;  /* Blue footer text */
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return onnxruntime.InferenceSession('model.onnx')

try:
    session = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

labels = {
    0: "No DR", 
    1: "Mild", 
    2: "Moderate", 
    3: "Severe", 
    4: "Proliferative DR"
}

# Image processing functions
def transform_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.5353, 0.3628, 0.2486], dtype=np.float32)
    std = np.array([0.2126, 0.1586, 0.1401], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict(input_img):
    if input_img is None:
        return None
    
    try:
        input_tensor = transform_image(input_img)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: input_tensor})[0][0]
        
        exp_preds = np.exp(prediction - np.max(prediction))
        probabilities = exp_preds / exp_preds.sum()
        
        confidences = {labels[i]: float(probabilities[i]) for i in labels}
        
        # Combine similar categories as in original code
        s = (confidences["Severe"] + confidences["Proliferative DR"])
        m = (confidences["Mild"] + confidences["Moderate"])
        
        filtered_confidences = {
            "No DR": confidences["No DR"],
            "Moderate": m,
            "Severe": s
        }
        
        return filtered_confidences
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return {"No DR": 0.33, "Moderate": 0.33, "Severe": 0.34}  # Return fallback values

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# Function to set example image directly
def set_example_image(example_index):
    path = f"sample/{example_index+1}.jpeg"
    if os.path.exists(path):
        st.session_state.image = Image.open(path)
        st.session_state.selected_example = example_index

    # Application title
st.markdown("<h1 style='text-align: center; color: #0D47A1 !important;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem; color: #1565C0 !important;' class='info-text'>Upload retinal images to detect signs of diabetic retinopathy</p>", unsafe_allow_html=True)

# Main columns layout
col1, col2 = st.columns([5, 7])

    # Left column - Input
with col1:
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0D47A1;'>Upload Image</h3>", unsafe_allow_html=True)
    
    # File uploader with instructions
    st.markdown("<p class='info-text'>Supported formats: JPEG, PNG</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # Handle file upload
    if uploaded_file is not None and (not hasattr(st.session_state, 'last_uploaded_file') or st.session_state.last_uploaded_file != uploaded_file.name):
        try:
            img = Image.open(uploaded_file)
            st.session_state.image = img
            st.session_state.last_uploaded_file = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    # Display the current input image
    if st.session_state.image is not None:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.image(st.session_state.image, width=300, caption="Selected Image")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Submit button
    if st.button("Analyze Image", use_container_width=True):
        if not model_loaded:
            st.error("Model not loaded. Please check your installation.")
        elif st.session_state.image is not None:
            with st.spinner("Analyzing image..."):
                st.session_state.results = predict(st.session_state.image)
        else:
            st.warning("Please upload an image or select an example first.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Examples section
    st.markdown("<div class='content-container' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0D47A1;'>Example Images</h3>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Click to use pre-loaded sample images</p>", unsafe_allow_html=True)
    
    # Load sample images
    sample_paths = ["sample/1.jpeg", "sample/2.jpeg", "sample/3.jpeg", "sample/4.jpeg"]
    example_cols = st.columns(4)
    
    # Display examples with click functionality
    for i, col in enumerate(example_cols):
        if os.path.exists(sample_paths[i]):
            img = Image.open(sample_paths[i])
            col.image(img, width=80, use_container_width=False)
            if col.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True, on_click=set_example_image, args=(i,)):
                pass  # The on_click callback handles setting the image
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Right column - Results
with col2:
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0D47A1;'>Analysis Results</h3>", unsafe_allow_html=True)
    
    if st.session_state.results:
        # Display top prediction with colors based on severity
        top_class = max(st.session_state.results, key=st.session_state.results.get)
        
        # Set color based on prediction
        if top_class == "No DR":
            result_color = "#4CAF50"  # Green
            severity = "Normal"
        elif top_class == "Moderate":
            result_color = "#FF9800"  # Amber
            severity = "Moderate"
        else:
            result_color = "#F44336"  # Red
            severity = "Severe"
        
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: {result_color} !important; font-size: 2rem;'>{severity}</h2>
            <p style='color: #1565C0 !important;'>Diabetic Retinopathy Classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        # Display bars for each class
        for label, confidence in st.session_state.results.items():
            percentage = int(confidence * 100)
            
            # Set bar color based on category
            if label == "No DR":
                bar_color = "linear-gradient(90deg, #4CAF50, #8BC34A)"
            elif label == "Moderate":
                bar_color = "linear-gradient(90deg, #FF9800, #FFCC80)"
            else:  # Severe
                bar_color = "linear-gradient(90deg, #F44336, #EF9A9A)"
                
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:15px">
                    <div style="width:100px;font-weight:500;color:#1A4B8C !important;">{label}</div>
                    <div style="flex-grow:1;background:#E0E0E0;border-radius:10px;margin:0 10px;overflow:hidden;">
                        <div style="width:{percentage}%;height:12px;background:{bar_color};border-radius:10px"></div>
                    </div>
                    <div style="width:50px;text-align:right;font-weight:500;color:#1A4B8C !important;">{percentage}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with st.container():
            st.markdown('<div class="flag-btn" style="margin-top: 10px;">', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder for results
        st.markdown("""
        <div style='
            height: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #F5F7FA;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px dashed #BDBDBD;
        '>
            <img src="https://cdn-icons-png.flaticon.com/512/7920/7920957.png" width="60" style="opacity: 0.5;">
            <p style='color: #1565C0; margin-top: 15px; font-weight: 500;'>Upload or select an example image<br>and click "Analyze Image" to view results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

