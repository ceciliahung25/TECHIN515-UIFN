import replicate
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import io
import base64
import re
import time
from hydralit import HydraApp, HydraHeadApp
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import pandas as pd
import json
from datetime import datetime
import os  # Add this line to import the os module
import pytz  # This module helps with timezone conversions

# Load environment variables
load_dotenv()

# Azure Blob Storage credentials
STORAGE_ACCOUNT_NAME = os.getenv('STORAGE_ACCOUNT_NAME')  # Read from environment variable
STORAGE_ACCOUNT_KEY = os.getenv('STORAGE_ACCOUNT_KEY')    # Read from environment variable
CONTAINER_NAME = 'cloud'  # Assuming 'cloud' is the container name

# Animal emojis mapping
animal_emojis = {
    "dog": "🐕", "bird": "🐦", "cat": "🐈", "elephant": "🐘", "fish": "🐟",
    "fox": "🦊", "horse": "🐎", "lion": "🦁", "monkey": "🐒", "mouse": "🐁",
    "owl": "🦉", "panda": "🐼", "rabbit": "🐇", "snake": "🐍", "tiger": "🐅",
    "unicorn": "🦄", "dragon": "🐉", "swan": "🦢", "cow": "🐄", "bear": "🐻"
}

# Function to inject JavaScript for getting the screen width
def get_screen_width():
    js = """
    <script>
    const width = window.innerWidth;
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: width}, '*');
    </script>
    """
    components.html(js, height=0, width=0)

# Define your functions and classes below
def get_blob_service_client(account_name, account_key):
    return BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=account_key
    )

def get_latest_blob_names(account_name, account_key, container_name, prefix, count=5):
    blob_service_client = get_blob_service_client(account_name, account_key)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = sorted(
        container_client.list_blobs(name_starts_with=prefix),
        key=lambda b: b.last_modified,
        reverse=True
    )
    latest_blobs = blobs[:count]
    return [blob.name for blob in latest_blobs]

def get_blob_data(account_name, account_key, container_name, blob_name):
    blob_service_client = get_blob_service_client(account_name, account_key)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()

def get_image_from_blob(account_name, account_key, container_name, blob_name):
    data = get_blob_data(account_name, account_key, container_name, blob_name)
    stream = io.BytesIO(data)
    return Image.open(stream)

def get_sensor_data_from_blobs(account_name, account_key, container_name, blob_names):
    data_frames = []
    for blob_name in blob_names:
        data = get_blob_data(account_name, account_key, container_name, blob_name)
        sensor_data = json.loads(data)
        df = pd.DataFrame([sensor_data])
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def local_image_to_data_url(image):
    img_byte_arr = io.BytesIO()
    try:
        image.save(img_byte_arr, format=image.format)
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None
    img_byte_arr.seek(0)
    try:
        encoded_string = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image to base64: {e}")
        return None
    mime_type = image.format.lower()
    return f"data:image/{mime_type};base64,{encoded_string}"

def process_analysis_text(text):
    st.write("Processing Text: ", text)
    pattern = re.compile(r"\d+\.\s+(\w+)\s+-\s+(\d+)%")
    matches = pattern.findall(text)
    st.write("Matches Found: ", matches)
    return [(match[0], int(match[1])) for match in matches]

def submit_analysis(image):
    image_url = local_image_to_data_url(image)
    if image_url is None:
        st.error("Failed to convert image to data URL.")
        return
    input_data = {
        "image": image_url,
        "prompt": "What are the top 5 animals this cloud looks like, with confidence scores in percentage? Please only give me the top 5 animals and their confidence scores in percentage. No other description!",
    }
    try:
        output_generator = replicate.run(
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            input=input_data,
        )
        output = list(output_generator)
        st.write("Debug: ", output)
        analysis_text = " ".join(output)
        st.write("Analysis Text: ", analysis_text)
        extracted_results = process_analysis_text(analysis_text)
        st.write("Extracted Results: ", extracted_results)
        if not extracted_results:
            extracted_results = [("Unknown", 0)] * 5
        st.session_state["extracted_results"] = extracted_results
        st.session_state["analysis_complete"] = True
        st.experimental_rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Unable to generate the riddle. Please check your Replicate credentials.")

def get_blob_metadata(account_name, account_key, container_name, blob_name):
    blob_service_client = get_blob_service_client(account_name, account_key)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_properties = blob_client.get_blob_properties()
    metadata = blob_properties.metadata or {}
    time_taken = metadata.get('time_taken', 'Unknown time')
    location = metadata.get('location', 'Unknown location')
    return {'time_taken': time_taken, 'location': location}

class CloudRiddleApp(HydraHeadApp):
    def run(self):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state["page"] = "Landing Page"
            st.toast("A new cloud is available ☁️")
            time.sleep(4)

        if st.session_state["page"] == "Landing Page":
            if st.button("👀 Check my new cloud ☁️"):
                st.session_state["page"] = "Image Display"
                st.experimental_rerun()

        elif st.session_state["page"] == "Image Display":
            latest_image_blob_name = get_latest_blob_names(
                STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, 'photo_', count=1
            )[0]
            image = get_image_from_blob(
                STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, latest_image_blob_name
            )
            st.image(image, caption=f'Latest Image: {latest_image_blob_name}', width=800)
            st.session_state["latest_image"] = image

            latest_sensor_blob_names = get_latest_blob_names(
                STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, 'sensor_data_', count=5
            )
            sensor_data_df = get_sensor_data_from_blobs(
                STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, latest_sensor_blob_names
            )

            st.subheader("Latest Sensor Data")
            st.write(sensor_data_df)

            st.subheader("Sensor Data Plot")
            for col in sensor_data_df.columns:
                sensor_data_df[col] = pd.to_numeric(sensor_data_df[col], errors='coerce')
            st.line_chart(sensor_data_df)

            if st.button("Confirm and Reveal the Riddle"):
                st.session_state["page"] = "Riddle Reveal"
                st.experimental_rerun()

        elif st.session_state["page"] == "Riddle Reveal":
            if "latest_image" not in st.session_state or st.session_state["latest_image"] is None:
                st.warning("Unable to fetch the image. Returning to the 'Image Display' page.")
                st.session_state["page"] = "Image Display"
                st.experimental_rerun()
            else:
                image = st.session_state["latest_image"]
                st.image(image, caption="Latest Cloud Photo", width=800)

                if "user_response" not in st.session_state:
                    st.session_state["user_response"] = ""
                st.session_state["user_response"] = st.text_input(
                    "What objects do you think the cloud looks like?",
                    value=st.session_state["user_response"]
                )
                if st.button("Submit and Reveal the Riddle"):
                    submit_analysis(image)

                if "analysis_complete" in st.session_state and st.session_state["analysis_complete"]:
                    extracted_results = st.session_state["extracted_results"]
                    st.subheader("Top 5 Similarities:")
                    for index, (item, similarity) in enumerate(extracted_results, start=1):
                        emoji = animal_emojis.get(item.lower(), "")
                        st.write(f"{index}. {emoji} {item}: {similarity}%")

                    if st.button("Check Next Cloud ☁️"):
                        st.session_state["page"] = "Image Display"
                        st.session_state.pop("latest_image", None)
                        st.session_state.pop("extracted_results", None)
                        st.session_state.pop("user_response", None)
                        st.session_state.pop("analysis_complete", None)
                        st.experimental_rerun()

class TimeLapseApp(HydraHeadApp):
    def run(self):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.title("Time-Lapse Photography")

        # Call to set the screen width in session state
        get_screen_width()

        # Check if image details are being shown
        if "show_details" not in st.session_state:
            st.session_state["show_details"] = False
            st.session_state["selected_image"] = None

        if not st.session_state["show_details"]:
            # Fetch the latest images
            latest_images = get_latest_blob_names(
                STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, 'photo_', count=9
            )

            # Determine the number of columns based on the screen width
            if st.session_state.get("screen_width", 800) < 800:
                cols_per_row = 2
            else:
                cols_per_row = 3

            cols = st.columns(cols_per_row)
            for index, image_name in enumerate(latest_images):
                col = cols[index % cols_per_row]
                with col:
                    image = get_image_from_blob(
                        STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, image_name
                    )
                    st.image(image, caption=f'Image {index+1}', width=100)
                    if st.button(f"View Details {index+1}", key=f'detail-{index+1}'):
                        st.session_state["selected_image"] = image_name
                        st.session_state["show_details"] = True
                        st.experimental_rerun()
        else:
            image_name = st.session_state["selected_image"]
            metadata = get_blob_metadata(STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, image_name)
            image = get_image_from_blob(STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY, CONTAINER_NAME, image_name)
            st.image(image, caption="Detailed View", use_column_width=True)
            
            # Use the new function to get the date and time from the filename
            date_time_taken = extract_datetime_from_filename(image_name)
            st.write(f"Time Taken: {date_time_taken}")
            
            if st.button("Back to album"):
                st.session_state["show_details"] = False
                st.session_state["selected_image"] = None
                st.experimental_rerun()

def extract_datetime_from_filename(filename):
    # Extract the timestamp part from the filename which seems to be in 'yyyyMMdd' or 'yyyyMMdd_HHmmss' format
    timestamp_str = filename.split('_')[1].split('.')[0]

    try:
        # Try to parse with time information first
        dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    except ValueError:
        # If fails, try to parse without time information
        dt = datetime.strptime(timestamp_str, '%Y%m%d')

    # Convert to a more readable format
    return dt.strftime('%Y-%m-%d %H:%M:%S') if 'H' in dt.strftime('%Y-%m-%d %H:%M:%S') else dt.strftime('%Y-%m-%d')

if __name__ == "__main__":
    app = HydraApp(
        title="Cloud Riddle and Time-Lapse",
        favicon="🌤️",
        use_navbar=True,
        navbar_sticky=False,
    )
    app.add_app("Cloud Riddle", icon="☁️", app=CloudRiddleApp())
    app.add_app("Time-Lapse", icon="⏳", app=TimeLapseApp())
    app.run()
