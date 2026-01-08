import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from scipy.ndimage import binary_fill_holes

# --- 1. SAFE IMPORTS ---
try:
    from transformers import pipeline
except ImportError:
    st.error("Transformers library missing. Run: pip install transformers torch")
    st.stop()

# --- 2. SETUP ---
st.set_page_config(layout="wide", page_title="BG remover")

@st.cache_resource
def load_depth_model():
    """Loads Depth Model."""
    device = 0 if torch.cuda.is_available() else -1
    model_name = "depth-anything/Depth-Anything-V2-Small-hf" 
    return pipeline(
        task="depth-estimation", 
        model=model_name, 
        use_fast=True, 
        device=device
    )

depth_estimator = load_depth_model()

# --- 3. HELPER FUNCTIONS ---

def get_local_variance(image, ksize=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_float = gray.astype(np.float32)
    img_sq = img_float ** 2
    mu = cv2.blur(img_float, (ksize, ksize))
    mu_sq = cv2.blur(img_sq, (ksize, ksize))
    variance = mu_sq - (mu ** 2)
    sigma = np.sqrt(np.abs(variance))
    sigma_norm = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sigma_norm

def get_auto_threshold(roughness_map, method="triangle"):
    if method == "triangle":
        thresh_val, _ = cv2.threshold(roughness_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return thresh_val
    # elif method == "otsu":
    #     thresh_val, _ = cv2.threshold(roughness_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     return thresh_val
    return 20

def fusion_remove_hand(image, depth_mask, hsv_settings, texture_thresh_val):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l1 = np.array([0, hsv_settings['s_min'], hsv_settings['v_min']])
    u1 = np.array([hsv_settings['h1_max'], 255, 255])
    l2 = np.array([hsv_settings['h2_min'], hsv_settings['s_min'], hsv_settings['v_min']])
    u2 = np.array([180, 255, 255])
    skin_mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    
    roughness_map = get_local_variance(image, ksize=7)
    _, smooth_mask = cv2.threshold(roughness_map, texture_thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    confirmed_hand = cv2.bitwise_and(skin_mask, smooth_mask)
    confirmed_hand = cv2.morphologyEx(confirmed_hand, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    confirmed_hand = cv2.dilate(confirmed_hand, np.ones((3,3), np.uint8), iterations=1)
    
    clean_mask = cv2.bitwise_and(depth_mask, cv2.bitwise_not(confirmed_hand))
    return clean_mask, confirmed_hand

def connect_boundary_endpoints(mask, morph_kernel_size=10, line_thickness=1):
    """
    Connects broken boundaries by finding extreme points of contours
    and drawing lines between closest endpoints. Finally fills holes using scipy.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    morph = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask, mask
    
    connected_mask = morph.copy()
    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(contours)):
        min_dist = max(mask.shape[0], mask.shape[1])
        closest_line = []
        
        ci = contours[i]
        ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
        ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
        ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
        ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
        ci_list = [ci_bottom, ci_left, ci_right, ci_top]
        
        for pt in ci_list:
            cv2.circle(debug_img, pt, 4, (0, 255, 255), -1)
        
        for j in range(i + 1, len(contours)):
            cj = contours[j]
            cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
            cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
            cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
            cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
            cj_list = [cj_bottom, cj_left, cj_right, cj_top]
            
            for pt1 in ci_list:
                for pt2 in cj_list:
                    dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
                    if dist < min_dist:
                        min_dist = dist
                        closest_line = [pt1, pt2, min_dist]
        
        if len(closest_line) > 0:
            cv2.line(connected_mask, closest_line[0], closest_line[1], 255, thickness=line_thickness)
            cv2.line(debug_img, closest_line[0], closest_line[1], (0, 255, 0), thickness=2)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_close)
    
    filled_mask = binary_fill_holes(connected_mask).astype(np.uint8) * 255
    
    return filled_mask, debug_img

def extract_largest_object(mask):
    """Just extracts the largest connected component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(mask)
    cv2.drawContours(result, [largest], -1, 255, thickness=-1)
    
    return result

def apply_mask_to_color(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

# --- 4. MAIN UI ---

def main():
    st.title("🔬 Boundary Endpoint Connection + Depth AND")
    st.markdown("Final output = Connected Boundary AND Depth Mask")

    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.header("⚙️ Controls")
        
        with st.expander("Threshold Settings", expanded=True):
            thresh_mode = st.radio("Mode", ["Auto (Triangle)", "Auto (Otsu)", "Manual"])
            manual_thresh = 15
            if "Manual" in thresh_mode:
                manual_thresh = st.slider("Value", 5, 100, 15)

        with st.expander("Endpoint Connection", expanded=True):
            enable_connection = st.checkbox("Enable Connection", value=True)
            morph_kernel = st.slider("Dilation Kernel", 5, 35, 19, step=2)
            line_thickness = st.slider("Line Thickness", 1, 15, 5)

        with st.expander("Advanced: Color (HSV)", expanded=False):
            hsv_settings = {'h1_max': 25, 'h2_min': 95, 's_min': 0, 'v_min': 0}
            st.write(hsv_settings)

    # --- FILE UPLOADER ---
    uploaded_files = st.file_uploader("Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # --- PROCESSING ---
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                # STEP 1: Depth
                depth_raw = depth_estimator(Image.fromarray(img_rgb))['depth']
                depth_map = np.array(depth_raw.resize((w, h)))
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, depth_mask = cv2.threshold(depth_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                depth_colored = apply_mask_to_color(img_bgr, depth_mask)
                
                # STEP 2: Threshold
                if "Auto" in thresh_mode:
                    temp_rough = get_local_variance(img_bgr, ksize=7)
                    method = "triangle" if "Triangle" in thresh_mode else "otsu"
                    thresh_val = get_auto_threshold(temp_rough, method=method)
                else:
                    thresh_val = manual_thresh
                
                # STEP 3: Hand Removal
                clean_mask, hand_mask = fusion_remove_hand(img_bgr, depth_mask, hsv_settings, thresh_val)
                clean_colored = apply_mask_to_color(img_bgr, clean_mask)
                hand_colored = apply_mask_to_color(img_bgr, hand_mask)
                
                # STEP 4A: Extract largest object first
                largest_only = extract_largest_object(clean_mask)
                largest_colored = apply_mask_to_color(img_bgr, largest_only)
                
                # STEP 4B: Apply Endpoint Connection
                if enable_connection:
                    connected_boundary, debug_viz = connect_boundary_endpoints(
                        largest_only,
                        morph_kernel_size=morph_kernel,
                        line_thickness=line_thickness
                    )
                    object_boundary = connected_boundary
                    method_used = "Endpoint Connection (on cleaned mask)"
                else:
                    object_boundary = largest_only
                    debug_viz = cv2.cvtColor(largest_only, cv2.COLOR_GRAY2BGR)
                    method_used = "No Connection (Largest Component Only)"
                
                boundary_colored = apply_mask_to_color(img_bgr, object_boundary)
                
                # STEP 5: NEW - Final output = Connected Boundary AND Depth Mask
                final_mask = cv2.bitwise_and(object_boundary, depth_mask)
                final_colored_output = apply_mask_to_color(img_bgr, final_mask)

            # --- INSPECTION UI ---
            st.markdown(f"### 📂 {uploaded_file.name}")
            
            tab_overview, tab_depth, tab_hand, tab_boundary, tab_final = st.tabs([
                "👁️ Overview", 
                "1️⃣ Depth Analysis", 
                "2️⃣ Hand Removal", 
                "3️⃣ Endpoint Connection",
                "4️⃣ Final AND"
            ])

            # TAB 1: OVERVIEW
            with tab_overview:
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original Input", use_container_width=True)
                c2.image(cv2.cvtColor(final_colored_output, cv2.COLOR_BGR2RGB), 
                        caption="Final Output (Boundary AND Depth)", use_container_width=True)
                
                st.info(f"**Method:** {method_used} + Depth AND Operation")

            # TAB 2: DEPTH
            with tab_depth:
                c1, c2, c3 = st.columns([1, 1, 1])
                c1.markdown("**Depth Map**")
                c1.image(depth_norm, use_container_width=True)
                c2.markdown("**Depth Mask**")
                c2.image(depth_mask, use_container_width=True)
                c3.markdown("**Depth Color**")
                c3.image(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB), use_container_width=True)

            # TAB 3: HAND REMOVAL
            with tab_hand:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**:red[Removed (Hand/Noise)]**")
                    st.image(cv2.cvtColor(hand_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown("**:green[Cleaned]**")
                    st.image(cv2.cvtColor(clean_colored, cv2.COLOR_BGR2RGB), use_container_width=True)

            # TAB 4: ENDPOINT CONNECTION
            with tab_boundary:
                st.markdown("**Step 1:** Extract largest → **Step 2:** Connect endpoints → **Step 3:** Fill holes")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Largest Only**")
                    st.image(largest_only, use_container_width=True)
                with c2:
                    st.markdown("**Debug: Connections**")
                    st.image(debug_viz, use_container_width=True, channels="BGR")
                    st.caption("Yellow = extreme points, Green = connecting lines")
                with c3:
                    st.markdown("**Connected + Filled**")
                    st.image(object_boundary, use_container_width=True)

            # TAB 5: NEW - FINAL AND OPERATION
            with tab_final:
                st.markdown("**Final Step:** Apply AND operation between Connected Boundary and Depth Mask")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Connected Boundary**")
                    st.image(object_boundary, use_container_width=True)
                    st.image(cv2.cvtColor(boundary_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown("**Depth Mask**")
                    st.image(depth_mask, use_container_width=True)
                    st.image(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c3:
                    st.markdown("**Final AND Result**")
                    st.image(final_mask, use_container_width=True)
                    st.image(cv2.cvtColor(final_colored_output, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.success("✅ This removes any filled areas that depth says should be background (holes, gaps)")

            st.divider()

if __name__ == "__main__":
    main()