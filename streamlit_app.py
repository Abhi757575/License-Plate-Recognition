import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from sort.sort.sort import Sort
from ultralytics import YOLO

from util import get_car, read_license_plate

COCO_VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

VIDEO_PAGE = "Video LPR"
IMAGE_PAGE = "Image LPR"


def _format_bbox(values: List[float]) -> str:
    """Convert bounding box coordinates to a compact string."""
    return "[{} {} {} {}]".format(*(int(round(v)) for v in values))


@st.cache_resource
def load_models() -> Tuple[YOLO, YOLO]:
    """Load detection models once per Streamlit session."""
    coco_model = YOLO("yolov8n.pt")
    license_plate_detector = YOLO("models/license_plate_detector.pt")
    return coco_model, license_plate_detector


class ProgressCallbacks:
    """Utility to capture Streamlit callbacks for the video pipeline."""

    def __init__(self, progress_bar: st.delta_generator.DeltaGenerator, status: st.delta_generator.DeltaGenerator):
        self.progress_bar = progress_bar
        self.status = status

    def update_progress(self, value: float) -> None:
        self.progress_bar.progress(value)

    def update_status(self, message: str) -> None:
        self.status.text(message)


def analyze_video(
    video_path: str,
    coco_model: YOLO,
    license_plate_model: YOLO,
    vehicle_classes: List[int],
    confidence_threshold: float,
    max_frames: Optional[int] = None,
    output_video_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[
    Dict[int, Dict[int, dict]],
    Optional[np.ndarray],
    List[dict],
    Dict[str, int],
]:
    """Run the YOLO+SORT pipeline and collect annotated results for a video."""

    tracker = Sort()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    last_annotated = None
    results: Dict[int, Dict[int, dict]] = {}
    entries: List[dict] = []
    unique_plates = set()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_writer: Optional[imageio.Writer] = None
    video_writer_error: Optional[str] = None

    if output_video_path and frame_width > 0 and frame_height > 0:
        try:
            video_writer = imageio.get_writer(
                output_video_path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                bitrate="2000k",
                ffmpeg_log_level="error",
            )
        except Exception as exc:  # pragma: no cover - codec availability varies
            video_writer_error = str(exc)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_index >= max_frames:
                break

            detections = coco_model(frame)[0]
            detections_for_tracking = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if score < confidence_threshold:
                    continue
                if int(class_id) not in vehicle_classes:
                    continue
                detections_for_tracking.append([x1, y1, x2, y2, score])

            tracking_input = (
                np.array(detections_for_tracking)
                if detections_for_tracking
                else np.zeros((0, 5))
            )
            track_ids = tracker.update(tracking_input)
            annotated = frame.copy()

            license_plates = license_plate_model(frame)[0]
            frame_results: Dict[int, dict] = {}
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                if score < confidence_threshold:
                    continue

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id == 0:
                    continue

                x1_i, y1_i = max(0, int(round(x1))), max(0, int(round(y1)))
                x2_i, y2_i = min(frame.shape[1], int(round(x2))), min(frame.shape[0], int(round(y2)))
                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                license_plate_crop = frame[y1_i:y2_i, x1_i:x2_i]
                if license_plate_crop.size == 0:
                    continue

                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(
                    license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV
                )

                license_plate_text, text_score = read_license_plate(license_plate_thresh)
                if not license_plate_text:
                    continue

                frame_results[int(car_id)] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": score,
                        "text_score": text_score,
                    },
                }

                entries.append(
                    {
                        "frame_nmr": frame_index,
                        "car_id": int(car_id),
                        "car_bbox": _format_bbox([xcar1, ycar1, xcar2, ycar2]),
                        "license_plate_bbox": _format_bbox([x1, y1, x2, y2]),
                        "license_plate_bbox_score": round(score, 3),
                        "license_number": license_plate_text,
                        "license_number_score": round(text_score, 3),
                    }
                )
                unique_plates.add(license_plate_text)

                cv2.rectangle(
                    annotated,
                    (int(round(xcar1)), int(round(ycar1))),
                    (int(round(xcar2)), int(round(ycar2))),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"ID:{int(car_id)}",
                    (int(round(xcar1)), max(0, int(round(ycar1))) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    annotated,
                    (x1_i, y1_i),
                    (x2_i, y2_i),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    license_plate_text,
                    (x1_i, max(0, y1_i) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if frame_results:
                results[frame_index] = frame_results

            last_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if video_writer is not None:
                video_writer.append_data(last_annotated)

            completed = frame_index + 1
            if progress_callback and total_frames:
                progress_callback(min(completed / total_frames, 1.0))
            if status_callback:
                status_callback(f"Analyzed frame {completed}/{total_frames or '?'}")

            frame_index += 1

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.close()

    summary = {
        "frames_processed": frame_index,
        "license_plates_found": len(entries),
        "unique_plates": len(unique_plates),
        "video_writer_error": video_writer_error,
    }

    return results, last_annotated, entries, summary


def analyze_image(
    image: np.ndarray,
    coco_model: YOLO,
    license_plate_model: YOLO,
    vehicle_classes: List[int],
    confidence_threshold: float,
) -> Tuple[np.ndarray, List[dict], Dict[str, int]]:
    """Analyze a single image for license plates and annotate the detections."""

    annotated = image.copy()
    detections = coco_model(image)[0]
    car_regions: List[List[float]] = []
    entries: List[dict] = []
    unique_plates = set()
    car_id_sequence = 1

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score < confidence_threshold:
            continue
        if int(class_id) not in vehicle_classes:
            continue
        car_regions.append([x1, y1, x2, y2, car_id_sequence])
        car_id_sequence += 1

    license_plates = license_plate_model(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        if score < confidence_threshold:
            continue

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, car_regions)
        if car_id == 0:
            continue

        x1_i, y1_i = max(0, int(round(x1))), max(0, int(round(y1)))
        x2_i, y2_i = min(image.shape[1], int(round(x2))), min(image.shape[0], int(round(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        plate_crop = image[y1_i:y2_i, x1_i:x2_i]
        if plate_crop.size == 0:
            continue

        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        plate_text, text_score = read_license_plate(plate_thresh)
        if not plate_text:
            continue

        entries.append(
            {
                "frame_nmr": 0,
                "car_id": int(car_id),
                "car_bbox": _format_bbox([xcar1, ycar1, xcar2, ycar2]),
                "license_plate_bbox": _format_bbox([x1, y1, x2, y2]),
                "license_plate_bbox_score": round(score, 3),
                "license_number": plate_text,
                "license_number_score": round(text_score, 3),
            }
        )
        unique_plates.add(plate_text)

        xcar1_i, ycar1_i = max(0, int(round(xcar1))), max(0, int(round(ycar1)))
        xcar2_i, ycar2_i = min(image.shape[1], int(round(xcar2))), min(image.shape[0], int(round(ycar2)))

        cv2.rectangle(
            annotated,
            (xcar1_i, ycar1_i),
            (xcar2_i, ycar2_i),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"ID:{int(car_id)}",
            (xcar1_i, max(0, ycar1_i) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            annotated,
            (x1_i, y1_i),
            (x2_i, y2_i),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            plate_text,
            (x1_i, max(0, y1_i) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    summary = {
        "frames_processed": 1,
        "license_plates_found": len(entries),
        "unique_plates": len(unique_plates),
    }

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, entries, summary


def _clear_uploaded_video() -> None:
    """Remove any cached uploaded video from the session and filesystem."""

    path = st.session_state.pop("uploaded_video_path", None)
    if path and os.path.exists(path):
        os.unlink(path)
    st.session_state.pop("uploaded_video_name", None)


def video_page(coco_model: YOLO, license_plate_model: YOLO) -> None:
    st.header("Video License Plate Recognition")
    st.markdown(
        "Process a recorded video with YOLO+SORT to detect vehicles and read license plates frame by frame."
    )

    with st.expander("Video source & detection settings", expanded=True):
        source = st.radio(
            "Video source",
            ("Sample video", "Upload video"),
            horizontal=True,
            key="video_source_radio",
        )

        video_path = None
        if source == "Sample video":
            sample_path = os.path.join(os.getcwd(), "sample.mp4")
            if os.path.exists(sample_path):
                video_path = sample_path
                st.success("Analyzing built-in sample.mp4")
            else:
                st.error("sample.mp4 is missing from the repository.")
            _clear_uploaded_video()
        else:
            upload = st.file_uploader(
                "Upload MP4 / AVI / MKV",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_upload",
            )
            if upload is not None:
                previous = st.session_state.get("uploaded_video_path")
                if previous and os.path.exists(previous):
                    os.unlink(previous)
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(upload.name)[1]
                )
                temp_file.write(upload.read())
                temp_file.flush()
                temp_file.close()
                st.session_state["uploaded_video_path"] = temp_file.name
                st.session_state["uploaded_video_name"] = upload.name
                video_path = temp_file.name
                st.success(f"Uploaded {upload.name} for analysis.")
            elif "uploaded_video_path" in st.session_state:
                video_path = st.session_state["uploaded_video_path"]
                st.info(
                    f"Ready to analyze {st.session_state.get('uploaded_video_name', 'the uploaded video')}."
                )

        vehicle_options = [f"{cid} - {name}" for cid, name in COCO_VEHICLE_CLASSES.items()]
        default_selection = vehicle_options
        allowed_classes = st.multiselect(
            "COCO vehicle classes to track",
            vehicle_options,
            default=default_selection,
        )
        selected_class_ids = [
            int(option.split(" - ")[0]) for option in allowed_classes if option
        ]
        if not selected_class_ids:
            selected_class_ids = list(COCO_VEHICLE_CLASSES.keys())

        confidence = st.slider(
            "Detection confidence",
            0.1,
            0.95,
            0.35,
            step=0.05,
            key="video_confidence_slider",
        )
        max_frames_input = st.number_input(
            "Max frames to process (0 = all)",
            min_value=0,
            value=0,
            step=1,
            key="max_frames_input",
        )

    analyze = st.button("Analyze video", key="analyze_video")
    if analyze:
        if not video_path:
            st.error("Provide a video before starting the analysis.")
            return

        annotated_video_path = None
        prev_annotated = st.session_state.get("annotated_video_path")
        if prev_annotated and os.path.exists(prev_annotated):
            os.unlink(prev_annotated)

        annotated_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        annotated_video_file.close()
        annotated_video_path = annotated_video_file.name

        progress = st.progress(0.0)
        status_placeholder = st.empty()
        callbacks = ProgressCallbacks(progress, status_placeholder)

        results, preview_frame, entries, summary = analyze_video(
            video_path,
            coco_model,
            license_plate_model,
            selected_class_ids,
            confidence,
            max_frames=max_frames_input or None,
            progress_callback=callbacks.update_progress,
            status_callback=callbacks.update_status,
            output_video_path=annotated_video_path,
        )

        st.session_state["annotated_video_path"] = annotated_video_path

        st.success("Video analysis completed.")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Frames processed", summary["frames_processed"])
        metric_cols[1].metric("Plates read", summary["license_plates_found"])
        metric_cols[2].metric("Unique plates", summary["unique_plates"])

        if entries:
            df = pd.DataFrame(entries)
            st.subheader("Recognized license plates")
            st.dataframe(df)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv_bytes,
                "video_lpr_results.csv",
                "text/csv",
            )
        else:
            st.info("No license plates were recognized in this video.")

        if preview_frame is not None:
            st.subheader("Last annotated frame")
            st.image(preview_frame, use_column_width=True)

        video_error = summary.get("video_writer_error")
        if video_error:
            st.warning(f"Annotated video could not be generated: {video_error}")
        elif (
            annotated_video_path
            and os.path.exists(annotated_video_path)
            and os.path.getsize(annotated_video_path) > 0
        ):
            st.subheader("Annotated video")
            st.video(annotated_video_path)
        else:
            st.warning("Annotated video could not be generated.")


def image_page(coco_model: YOLO, license_plate_model: YOLO) -> None:
    st.header("Image License Plate Recognition")
    st.markdown("Upload a single image to detect vehicles and read their license plates instantly.")

    with st.expander("Detection settings", expanded=True):
        vehicle_options = [f"{cid} - {name}" for cid, name in COCO_VEHICLE_CLASSES.items()]
        default_selection = vehicle_options
        allowed_classes = st.multiselect(
            "COCO vehicle classes to consider",
            vehicle_options,
            default=default_selection,
            key="image_vehicle_classes",
        )
        selected_class_ids = [
            int(option.split(" - ")[0]) for option in allowed_classes if option
        ]
        if not selected_class_ids:
            selected_class_ids = list(COCO_VEHICLE_CLASSES.keys())

        confidence = st.slider(
            "Detection confidence",
            0.1,
            0.95,
            0.35,
            step=0.05,
            key="image_confidence",
        )

    upload = st.file_uploader(
        "Upload JPG / PNG / WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    analyze = st.button("Analyze image", key="analyze_image")
    if analyze:
        if upload is None:
            st.error("Upload an image before running the image analysis.")
            return

        image_bytes = upload.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Unable to decode the uploaded image.")
            return

        annotated_image, entries, summary = analyze_image(
            image,
            coco_model,
            license_plate_model,
            selected_class_ids,
            confidence,
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric("Image processed", "1")
        metric_cols[1].metric("Plates read", summary["license_plates_found"])
        metric_cols[2].metric("Unique plates", summary["unique_plates"])

        if entries:
            df = pd.DataFrame(entries)
            st.subheader("Recognized license plates")
            st.dataframe(df)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv_bytes,
                "image_lpr_results.csv",
                "text/csv",
            )
        else:
            st.info("No license plates were recognized in this image.")

        st.subheader("Annotated image")
        st.image(annotated_image, use_column_width=True)


def main() -> None:
    st.set_page_config(page_title="License Plate Recognition", layout="wide")
    st.title("License Plate Recognition Toolkit")
    st.markdown("Use the tabs below to select between video and image LPR workflows.")

    coco_model, license_plate_model = load_models()

    tab_video, tab_image = st.tabs([VIDEO_PAGE, IMAGE_PAGE])
    with tab_video:
        video_page(coco_model, license_plate_model)
    with tab_image:
        image_page(coco_model, license_plate_model)


if __name__ == "__main__":
    main()
