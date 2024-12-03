import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QSizePolicy, QLineEdit, QDialog,
    QSpinBox, QProgressDialog, QProgressBar, QFrame, QSpacerItem
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QMovie
from PyQt5.QtCore import  QTimer, Qt, pyqtSlot, QPropertyAnimation, QThread, pyqtSignal, QObject, QCoreApplication
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from configs import ablation_configurations
import mediapipe as mp
from PIL import Image
import os
import time 
from process_dataset_module import construct_video, dynamic_select_frames, parse_frame_numbers
from extract_to_video_folders import process_data
from collections import OrderedDict
import dlib
from concurrent.futures import ThreadPoolExecutor

class HoverButton(QPushButton):
    """
    Custom QPushButton that changes color on hover.
    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.animate_hover(True)

    def leaveEvent(self, event):
        self.animate_hover(False)

    def animate_hover(self, hover):
        animation = QPropertyAnimation(self, b'background-color')
        animation.setDuration(200)
        if hover:
            animation.setEndValue(QColor('#45a049'))  # Darker green on hover
        else:
            animation.setEndValue(QColor('#4CAF50'))  # Original green
        animation.start()

class LoginDialog(QDialog):
    """
    Login dialog class with username and password fields.
    """
    def __init__(self):
        super(LoginDialog, self).__init__()
        self.setWindowTitle('Login')
        self.setFixedSize(400, 300)

        self.layout = QVBoxLayout(self)

        self.icon_label = QLabel(self)
        self.icon_label.setPixmap(QPixmap('media/icon.png').scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.username_label = QLabel('Username')
        self.username_input = QLineEdit(self)
        self.password_label = QLabel('Password')
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.login_button = QPushButton('Login', self)
        self.register_button = QPushButton('Register', self)

        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.register_button)

        self.login_button.clicked.connect(self.handle_login)
        self.register_button.clicked.connect(self.open_registration)

        self.accepted = False

    def handle_login(self):
        """
        Handles the login process. Replace with actual logic.
        """
        if self.username_input.text() == '' and self.password_input.text() == '':
            self.accepted = True
            self.accept()
        else:
            QMessageBox.warning(self, 'Error', 'Incorrect username or password')

    def open_registration(self):
        """
        Opens the registration dialog.
        """
        self.accepted = False
        self.reject()
        self.registration_dialog = RegistrationDialog()
        if self.registration_dialog.exec_() == QDialog.Accepted:
            self.username_input.setText(self.registration_dialog.username_input.text())
            self.password_input.setFocus()

    def reject(self):
        self.accepted = False
        super(LoginDialog, self).reject()

class RegistrationDialog(QDialog):
    """
    Registration dialog class with username and password fields.
    """
    def __init__(self):
        super(RegistrationDialog, self).__init__()
        self.setWindowTitle('Register')
        self.setFixedSize(400, 300)

        self.layout = QVBoxLayout(self)

        self.icon_label = QLabel(self)
        self.icon_label.setPixmap(QPixmap('icon.png').scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.username_label = QLabel('Username')
        self.username_input = QLineEdit(self)
        self.password_label = QLabel('Password')
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.register_button = QPushButton('Register', self)

        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.register_button)

        self.register_button.clicked.connect(self.handle_registration)

    def handle_registration(self):
        """
        Handles the registration process. Replace with actual logic.
        """
        QMessageBox.information(self, 'Success', 'Registration successful')
        self.accept()

    def reject(self):
        super(RegistrationDialog, self).reject()

# Dataset Extraction Worker class
class DatasetExtractionWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, excel_file_path, dataset_parent_dir, output_folder, dataset_name, part=None):
        super().__init__()
        self.excel_file_path = excel_file_path
        self.dataset_parent_dir = dataset_parent_dir
        self.output_folder = output_folder
        self.dataset_name = dataset_name
        self.part = part

    def process(self):
        # Your dataset extraction logic here
        try:
            process_data(self.excel_file_path, self.dataset_parent_dir, self.output_folder, self.dataset_name, self.part)
            self.finished.emit()
        except ValueError as e:
            QMessageBox.warning(None, 'Warning', str(e))
            self.finished.emit()

# Dataset Processing Worker class
class DatasetWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, source_folder, destination_folder, sparse_rate, window_size):
        super().__init__()
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.sparse_rate = sparse_rate
        self.window_size = window_size

    def process(self):
        dataset_name = os.path.basename(self.source_folder.rstrip('/'))
        parent_folder = self.destination_folder
        window_sparse_folder = f"WINDOW_{self.window_size}_SPARSE_{self.sparse_rate}"

        left_mirror = os.path.join(parent_folder, window_sparse_folder, f"L_ALL_{dataset_name}_DATASET")
        right_mirror = os.path.join(parent_folder, window_sparse_folder, f"R_ALL_{dataset_name}_DATASET")
        
        os.makedirs(left_mirror, exist_ok=True)
        os.makedirs(right_mirror, exist_ok=True)

        total_folders = self.count_total_folders()
        processed_folders = 0

        for class_name in os.listdir(self.source_folder):
            class_path = os.path.join(self.source_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            output_class_folder_left = os.path.join(left_mirror, class_name)
            output_class_folder_right = os.path.join(right_mirror, class_name)
            os.makedirs(output_class_folder_left, exist_ok=True)
            os.makedirs(output_class_folder_right, exist_ok=True)

            for video_folder_name in os.listdir(class_path):
                video_folder_path = os.path.join(class_path, video_folder_name)
                if not os.path.isdir(video_folder_path):
                    continue

                onset, apex, offset = parse_frame_numbers(video_folder_name)
                frames_to_use = dynamic_select_frames(onset, apex, offset, self.window_size, self.sparse_rate)
                if not frames_to_use:
                    continue

                output_video_name = f"{video_folder_name}.avi"
                left = f"_left_aug_{self.window_size}_{self.sparse_rate}.avi"
                right = f"_right_aug_{self.window_size}_{self.sparse_rate}.avi"
                augmented_output_video_name_left = video_folder_name + left
                augmented_output_video_name_right = video_folder_name + right

                output_path_original_left = os.path.join(output_class_folder_left, output_video_name)
                output_path_augmented_left = os.path.join(output_class_folder_left, augmented_output_video_name_left)
                output_path_original_right = os.path.join(output_class_folder_right, output_video_name)
                output_path_augmented_right = os.path.join(output_class_folder_right, augmented_output_video_name_right)

                try:
                    if os.path.exists(output_path_augmented_left) or os.path.exists(output_path_original_left):
                        continue
                    if os.path.exists(output_path_augmented_right) or os.path.exists(output_path_original_right):
                        continue

                    construct_video(video_folder_path, frames_to_use, output_path_original_left, output_path_augmented_left, "left")
                    construct_video(video_folder_path, frames_to_use, output_path_original_right, output_path_augmented_right, "right")
                    processed_folders += 1
                    progress_percent = int((processed_folders / total_folders) * 100)
                    self.progress.emit(progress_percent)
                except Exception as e:
                    processed_folders += 1
                    progress_percent = int((processed_folders / total_folders) * 100)
                    self.progress.emit(progress_percent)
                    continue              

        self.finished.emit()

    def count_total_folders(self):
        total_folders = 0
        for class_name in os.listdir(self.source_folder):
            class_path = os.path.join(self.source_folder, class_name)
            if os.path.isdir(class_path):
                for video_folder_name in os.listdir(class_path):
                    video_folder_path = os.path.join(class_path, video_folder_name)
                    if os.path.isdir(video_folder_path):
                        total_folders += 1
        return total_folders

# Main Window class
class DatasetProcessingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Dataset Processing')
        self.setMinimumSize(800, 450)  # Increased width for two panels
        with open('styles/styles.qss', 'r') as style_file:
            stylesheet = style_file.read()       
        self.setStyleSheet(stylesheet)  # Apply the stylesheet to the entire window
        self.init_ui()
        self.center()

    def init_ui(self):
        # Left Panel
        left_panel = QFrame(self)
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setFrameShadow(QFrame.Raised)

        left_heading = QLabel('Extract Raw Dataset Video Folders', self)
        left_heading.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.select_excel_button = QPushButton('Select Excel Annotation File', self)
        self.select_dataset_root_folder_button = QPushButton('Select Dataset Root Folder', self)
        self.select_destination_folder_button_left = QPushButton('Select Destination Folder', self)
        self.process_dataset_button_left = QPushButton('Process Dataset', self)
        
        self.excel_file_label = QLabel('Selected Excel Annotation File: None', self)
        self.dataset_root_folder_label = QLabel('Selected Dataset Root Folder: None', self)
        self.destination_folder_label_left = QLabel('Selected Destination Folder: None', self)
        
        self.dataset_name_input = QLineEdit(self)
        self.dataset_name_input.setPlaceholderText('Enter dataset name')

        # Left panel layout
        left_form_layout = QVBoxLayout()
        left_form_layout.addWidget(left_heading)
        left_form_layout.addWidget(self.excel_file_label)
        left_form_layout.addWidget(self.select_excel_button)
        left_form_layout.addWidget(self.dataset_root_folder_label)
        left_form_layout.addWidget(self.select_dataset_root_folder_button)
        left_form_layout.addWidget(self.destination_folder_label_left)
        left_form_layout.addWidget(self.select_destination_folder_button_left)
        left_form_layout.addWidget(self.dataset_name_input)
        left_form_layout.addWidget(self.process_dataset_button_left)

        left_panel.setLayout(left_form_layout)

        # Right Panel
        right_panel = QFrame(self)
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setFrameShadow(QFrame.Raised)

        right_heading = QLabel('Process Video Folders and Augment', self)
        right_heading.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.dataset_folder_label = QLabel('Selected Dataset Folder: None', self)
        self.destination_folder_label = QLabel('Output Folder: None', self)

        self.select_dataset_folder_button = QPushButton('Select Dataset Folder', self)
        self.select_destination_folder_button = QPushButton('Select Destination Folder', self)
        self.process_dataset_button_right = QPushButton('Process Dataset', self)
        
        self.sparse_rate_spinbox = QSpinBox(self)
        self.window_size_spinbox = QSpinBox(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)

        # Set up SpinBox ranges and default values
        self.sparse_rate_spinbox.setMinimum(0)
        self.sparse_rate_spinbox.setMaximum(100)
        self.sparse_rate_spinbox.setValue(1)

        self.window_size_spinbox.setMinimum(1)
        self.window_size_spinbox.setMaximum(100)
        self.window_size_spinbox.setValue(1)

        # Right panel layout
        right_form_layout = QVBoxLayout()
        right_form_layout.addWidget(right_heading)
        right_form_layout.addWidget(self.dataset_folder_label)
        right_form_layout.addWidget(self.select_dataset_folder_button)
        right_form_layout.addWidget(self.destination_folder_label)
        right_form_layout.addWidget(self.select_destination_folder_button)
        right_form_layout.addWidget(QLabel('Sparse Rate:', self))
        right_form_layout.addWidget(self.sparse_rate_spinbox)
        right_form_layout.addWidget(QLabel('Window Size:', self))
        right_form_layout.addWidget(self.window_size_spinbox)
        right_form_layout.addWidget(self.process_dataset_button_right)
        right_form_layout.addWidget(self.progress_bar)

        right_panel.setLayout(right_form_layout)

        # Main Layout
        main_layout = QVBoxLayout()
        panel_layout = QHBoxLayout()
        panel_layout.addWidget(left_panel)
        panel_layout.addWidget(right_panel)
        main_layout.addLayout(panel_layout)

        # Add spacer to push buttons to the bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)
        
        # Set consistent button size for all buttons
        button_width = 350
        button_height = 35

        # Apply fixed size to buttons
        self.select_excel_button.setFixedSize(button_width, button_height)
        self.select_dataset_root_folder_button.setFixedSize(button_width, button_height)
        self.select_destination_folder_button_left.setFixedSize(button_width, button_height)
        self.process_dataset_button_left.setFixedSize(button_width, button_height)
        self.select_dataset_folder_button.setFixedSize(button_width, button_height)
        self.select_destination_folder_button.setFixedSize(button_width, button_height)
        self.process_dataset_button_right.setFixedSize(button_width, button_height)

        # Create close button and center it at the bottom
        close_button = QPushButton('Close', self)
        close_button.setStyleSheet("font-size: 14px; height: 20px; width: 200px; background-color: #ff4d4d; color: white;")
        button_box = QHBoxLayout()
        button_box.addStretch()
        button_box.addWidget(close_button)
        button_box.addStretch()

        main_layout.addLayout(button_box)

        self.setLayout(main_layout)

        # Allow the window to resize based on content
        self.setMinimumSize(800, 450)

        # Connect signals and slots
        self.select_excel_button.clicked.connect(self.select_excel_file)
        self.select_dataset_root_folder_button.clicked.connect(self.select_dataset_root_folder)
        self.select_destination_folder_button_left.clicked.connect(self.select_destination_folder_left)
        self.process_dataset_button_left.clicked.connect(self.process_dataset_extraction)
        self.select_dataset_folder_button.clicked.connect(self.select_dataset_folder)
        self.select_destination_folder_button.clicked.connect(self.select_destination_folder)
        self.process_dataset_button_right.clicked.connect(self.process_dataset)
        close_button.clicked.connect(self.accept)

    def select_excel_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx)", options=options)
        if file_path:
            self.excel_file_path = file_path
            self.excel_file_label.setText(f'Selected Excel File: {os.path.basename(file_path)}')

    def select_dataset_root_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Root Folder", options=options)
        if folder_path:
            self.dataset_root_folder = folder_path
            self.dataset_root_folder_label.setText(f'Selected Dataset Root Folder: {os.path.basename(folder_path)}')

    def select_destination_folder_left(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Destination Folder", options=options)
        if folder_path:
            self.destination_folder_left = folder_path
            self.destination_folder_label_left.setText(f'Selected Destination Folder: {os.path.basename(folder_path)}')

    def select_dataset_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", options=options)
        if folder_path:
            self.dataset_folder = folder_path
            self.dataset_folder_label.setText(f'Selected Dataset Folder: {os.path.basename(folder_path)}')

    def select_destination_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Destination Folder", options=options)
        if folder_path:
            self.destination_folder = folder_path
            self.destination_folder_label.setText(f'Output Folder: {os.path.basename(folder_path)}')

    def process_dataset_extraction(self):
        excel_file_path = self.excel_file_path
        dataset_parent_dir = self.dataset_root_folder
        output_folder = self.destination_folder_left
        dataset_name = self.dataset_name_input.text().strip().lower()
        part = None  # Handle part selection if needed

        if not excel_file_path or not dataset_parent_dir or not output_folder or not dataset_name:
            QMessageBox.warning(self, 'Warning', 'Please provide all required inputs.')
            return

        self.thread = QThread()
        self.worker = DatasetExtractionWorker(excel_file_path, dataset_parent_dir, output_folder, dataset_name, part)
        
        self.worker.moveToThread(self.thread)
        
        # Connect signals for progress and completion
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_extraction_finished)
        
        # Make sure the thread is cleaned up after finishing
        self.worker.finished.connect(self.thread.quit)  # Quit the thread loop
        self.worker.finished.connect(self.worker.deleteLater)  # Cleanup the worker
        self.thread.finished.connect(self.thread.deleteLater)  # Cleanup the thread
        
        self.thread.started.connect(self.worker.process)
        self.thread.start()

    def process_dataset(self):
        source_folder = self.dataset_folder
        destination_folder = self.destination_folder
        sparse_rate = self.sparse_rate_spinbox.value()
        window_size = self.window_size_spinbox.value()

        if not source_folder or not destination_folder:
            QMessageBox.warning(self, 'Warning', 'Please select source and destination folders.')
            return

        self.thread = QThread()
        self.worker = DatasetWorker(source_folder, destination_folder, sparse_rate, window_size)
        
        self.worker.moveToThread(self.thread)
        
        # Connect signals for progress and completion
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        
        # Ensure the thread is cleaned up after finishing
        self.worker.finished.connect(self.thread.quit)  # Quit the thread loop
        self.worker.finished.connect(self.worker.deleteLater)  # Cleanup the worker
        self.thread.finished.connect(self.thread.deleteLater)  # Cleanup the thread
        
        self.thread.started.connect(self.worker.process)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def on_extraction_finished(self):
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, 'Success', 'Dataset extraction completed.')

    def on_processing_finished(self):
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, 'Success', 'Dataset processing completed.')

    def center(self):
        frame_geom = self.frameGeometry()
        screen_center = QApplication.desktop().screen().rect().center()
        frame_geom.moveCenter(screen_center)
        self.move(frame_geom.topLeft())

class RealTimeProcessor:
    def __init__(self, processing_interval=0.1):
        self.last_processed_time = time.time()
        self.processing_interval = processing_interval  # Time interval in seconds

    def should_process(self):
        current_time = time.time()
        if current_time - self.last_processed_time >= self.processing_interval:
            self.last_processed_time = current_time
            return True
        return False


class ModelLoader(QObject):
    progress_update = pyqtSignal(int)  # Signal for progress update
    load_finished = pyqtSignal(str)  # Signal when model loading is done

    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=1)  # Thread executor for async loading
        self._emotion_classifier = None  # Store the classifier model here

    def run(self):
        # Start the loading process in a separate thread
        self.executor.submit(self._load_model)

    def _load_model(self):
        try:
            # Simulated progress updates for steps before loading
            for i in range(5):  # Simulating different stages
                time.sleep(0.5)  # Simulate time spent in each stage
                self.progress_update.emit((i + 1) * 10)  # Emit progress updates

            # Load the checkpoint
            self.progress_update.emit(60)  # Emitting some progress for loading checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            checkpoint_name = self.checkpoint_path.split('/')[-1]

            # Get model config
            self.progress_update.emit(70)  # Emit progress for model configuration
            model_config = self.get_ablation_config(checkpoint_name)
            self._emotion_classifier = model_config

            # Load state dictionary
            self.progress_update.emit(85)  # Emitting progress for loading model weights
            if isinstance(checkpoint, OrderedDict):
                self._emotion_classifier.load_state_dict(checkpoint)
            elif 'model_state_dict' in checkpoint:
                self._emotion_classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("The checkpoint does not contain a valid state dictionary.")

            # Finalize the loading process
            self.progress_update.emit(100)  # Emit 100% progress on completion
            self._emotion_classifier.eval()  # Set model to evaluation mode

            # Emit the load finished signal
            self.load_finished.emit(checkpoint_name)

        except Exception as e:
            self.load_finished.emit(f'Error: {e}')  # Emit error signal if an exception occurs

    def get_ablation_config(self, checkpoint_name):
        ablation_tests = ablation_configurations()

        # Extract the key part from the checkpoint name
        config_key = checkpoint_name.split('_fold')[0]

        # Find and return the configuration
        if config_key in ablation_tests:
            return ablation_tests[config_key]['model']
        else:
            raise ValueError(f"No configuration found for checkpoint '{checkpoint_name}'")

    def get_emotion_classifier(self):
        return self._emotion_classifier
    
class MicroExpressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = dlib.get_frontal_face_detector()
        self.feature_extractor = None
        self.emotion_classifier = None
        self.loaded_model_name = None
        self.num_classes = 3

        self.video_path = None
        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_fps = 30
        self.is_video_playing = False
        self.current_frame = None
        self.total_frames = None
        self.current_frame_index = 0
        self.landmarks_over_time = []
        self.landmark_plot_counter = 0
        self.landmark_plot_interval = 5
        self.landmark_indices = [33, 263, 61, 291, 0, 17]
        self.frame_buffer = []
        self.stride = 1

        self.webcam = cv2.VideoCapture(0)
        self.webcam_timer = QTimer(self)
        self.is_webcam_running = False
        self.real_time_processor = RealTimeProcessor(processing_interval=0.1)

        # Configuration parameters
        self.use_head_mask = False
        self.dropout = 0.0
        self.num_frames = 1
        self.threads = []

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.8, min_tracking_confidence=0.8)

        self.load_ui()
        self.setup_ui()

    def load_ui(self):
        loadUi('ui/micro_expression_app.ui', self)

    def setup_ui(self):
        self.setWindowTitle('Micro-Expression Analysis')
        self.video_frame.setMinimumSize(900, 480)

        # Connect signals and slots
        self.start_stop_button.clicked.connect(self.start_stop_webcam)
        self.load_model_button.clicked.connect(self.load_model_checkpoint)
        self.load_video_button.clicked.connect(self.load_video)
        self.exit_button.clicked.connect(self.exit_application)
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.video_slider.sliderPressed.connect(self.slider_pressed)
        self.video_slider.sliderReleased.connect(self.slider_released)
        self.video_slider.valueChanged.connect(self.update_frame_rate)

        # Connect dataset processing signals
        self.process_dataset_button.clicked.connect(self.open_dataset_processing_window)

        # Set consistent font size and alignment for labels
        font = QFont()
        font.setPointSize(14)
        for label in self.findChildren(QLabel):
            label.setFont(font)
            label.setAlignment(Qt.AlignCenter)

        # Adjust size policy for buttons to ensure consistent size
        buttons = [
            self.start_stop_button, self.process_dataset_button, self.load_video_button, 
            self.exit_button, self.play_pause_button, self.stop_button]
        for button in buttons: 
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.exit_button.setStyleSheet("background-color: #ff4d4d;")
        self.setup_canvases()
        self.setup_additional_widgets()

        # Placeholder label setup
        self.placeholder_label = QLabel(self.video_frame)
        self.placeholder_label.setMinimumSize(900, 480)  # Adjust size for testing

        self.setup_main_layout()

        self.set_placeholder()

    def open_dataset_processing_window(self):
        # Stop video playback if it is currently running
        if self.is_video_playing:
            self.stop_video()
        
        # Close any existing video capture resources
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        # Safely stop webcam capture if it's running
        if self.is_webcam_running:
            self.stop_webcam()
        
        # Error handling for opening the dataset processing window
        try:
            self.dataset_processing_window = DatasetProcessingWindow(self)
            self.dataset_processing_window.show()
        except Exception as e:
            print(f"Error occurred while opening dataset processing window: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open dataset processing window: {e}")

    def setup_canvases(self):
        # Create canvases and axes for different plots
        self.trend_canvas, self.trend_ax = self.create_canvas_with_ax('Emotion Trend Over Time', 400, 300)
        self.landmark_canvas, self.landmark_ax = self.create_canvas_with_ax('Landmark Trends Over Time', 400, 250)
        self.bar_canvas, self.bar_ax = self.create_canvas_with_ax('Emotion Analysis Bar Plot', 350, 300)
        self.pie_canvas, self.pie_ax = self.create_canvas_with_ax('Emotion Distribution', 350, 300)

        # Add canvases to your layout (graph_layout assumed to be defined)
        self.graph_layout.addWidget(self.trend_canvas)
        self.graph_layout.addWidget(self.landmark_canvas)

        row2_layout = QHBoxLayout()
        row2_layout.addWidget(self.pie_canvas)
        row2_layout.addWidget(self.bar_canvas)
        self.graph_layout.addLayout(row2_layout)


    def create_canvas_with_ax(self, title, min_width, min_height, xlabel='', ylabel=''):
        canvas = FigureCanvas(Figure(figsize=(5, 3)))
        ax = canvas.figure.subplots()
        ax.set_title(title, pad=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumSize(min_width, min_height)
        return canvas, ax

    def setup_additional_widgets(self):

        self.use_head_mask_checkbox.setChecked(self.use_head_mask)
        self.use_head_mask_checkbox.toggled.connect(self.update_use_head_mask)

        self.num_classes_spinbox.setMinimum(1)
        self.num_classes_spinbox.setMaximum(100)
        self.num_classes_spinbox.setValue(self.num_classes)
        self.num_classes_spinbox.valueChanged.connect(self.update_num_classes)

        self.num_frames_spinbox.setMinimum(1)
        self.num_frames_spinbox.setMaximum(100)
        self.num_frames_spinbox.setValue(self.num_frames)
        self.num_frames_spinbox.valueChanged.connect(self.update_num_frames)

        self.stride_spinbox.setMinimum(1)
        self.stride_spinbox.setMaximum(100)
        self.stride_spinbox.setValue(self.stride)
        self.stride_spinbox.valueChanged.connect(self.update_stride)

        self.update_input.setText(', '.join(map(str, self.landmark_indices)))
        self.update_input.textChanged.connect(self.update_landmark_indices)

    def setup_main_layout(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.addLayout(self.verticalLayout)
        self.main_layout.addLayout(self.graph_layout)
        self.setLayout(self.main_layout)

    def update_num_frames(self, value):
        self.num_frames = value

    def update_stride(self, value):
        self.stride = value

    def update_use_head_mask(self, checked):
        self.use_head_mask = checked

    def update_num_classes(self, value):
        self.num_classes = value

    def update_landmark_indices(self, input_text):
        print(f"Input text changed: {input_text}")  # Debug print to check signal activation
        input_text = input_text.strip()
        if input_text:
            try:
                self.landmark_indices = [int(idx.strip()) for idx in input_text.split(',') if idx.strip().isdigit()]
                print(f"Updated landmark indices: {self.landmark_indices}")  # Debug print to check parsing
                self.visualize_landmarks_over_time()  # Call the visualization method to update the plot
            except ValueError as e:
                print(f"Error parsing input: {e}")

    def set_placeholder(self):
        movie = QMovie('media/placeholder.gif')
        self.placeholder_label.setMovie(movie)
        movie.start()
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setVisible(not (self.is_video_playing or self.video_path or self.is_webcam_running))

    def load_model_checkpoint(self):
        checkpoint_path, _ = QFileDialog.getOpenFileName(self, 'Open Model Checkpoint', '', 'Model Checkpoint (*.pth)')
        if checkpoint_path:
            progress_dialog = QProgressDialog('Loading Model...', 'Cancel', 0, 100, self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setWindowTitle('Please Wait')

            # Create the thread and the model loader worker
            load_thread = QThread(self)
            self.model_loader = ModelLoader(checkpoint_path, self.device)

            print(self.model_loader)
            
            self.model_loader.moveToThread(load_thread)

            # Connect signals
            self.model_loader.progress_update.connect(progress_dialog.setValue)
            self.model_loader.load_finished.connect(lambda name: self.on_load_finished(name, progress_dialog))

            # Start the loading thread
            load_thread.started.connect(self.model_loader.run)
            load_thread.start()

            # Show the progress dialog
            progress_dialog.exec_()

            # Ensure thread cleanup after it's finished
            load_thread.finished.connect(load_thread.deleteLater)

        else:
            QMessageBox.warning(self, 'Warning', 'No checkpoint file selected.')


    def on_load_finished(self, checkpoint_name, progress_dialog):
        progress_dialog.close()  # Close the progress dialog as soon as loading is complete
        QCoreApplication.processEvents()  # Ensure the event loop processes the close event immediately

        if checkpoint_name.startswith('Error:'):
            QMessageBox.warning(self, 'Warning', checkpoint_name)
        else:
            self.loaded_model_name = checkpoint_name
            self.loaded_model_label.setText(self.loaded_model_name)
            QMessageBox.information(self, 'Success', f'Model "{self.loaded_model_name}" loaded successfully!')
            self.emotion_classifier = self.model_loader.get_emotion_classifier()

    def play_pause_video(self):
        if self.video_path is None:
            QMessageBox.warning(self, 'Warning', 'No video selected.')
            return
        if not self.is_video_playing and self.video_capture is None:
            QMessageBox.warning(self, 'Warning', 'No video loaded.')
            return
        if self.is_video_playing:
            self.video_timer.stop()
            self.play_pause_button.setText('Play')
            self.is_video_playing = False
        else:
            self.video_timer.start(1000 // self.video_fps)
            self.play_pause_button.setText('Pause')
            self.is_video_playing = True

    def stop_video(self):
        if self.video_capture is None:
            QMessageBox.warning(self, 'Warning', 'No video loaded.')
            return

        self.video_timer.stop()
        self.video_capture.release()
        self.video_frame.clear()
        self.frame_buffer = []
        self.play_pause_button.setText('Play')
        self.is_video_playing = False
        self.video_path = False
        self.set_placeholder()
        self.video_capture = None

    def exit_application(self):
        self.close()

    def process_rgb_frame(self, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        face_batches = []
        points_batch = []
        cropped_images = []
        landmarks_batch = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                head_outline_indices = [
                    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
                ]

                h, w, _ = rgb_frame.shape
                points = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in head_outline_indices]
                transform = transforms.Compose([
                    transforms.Resize((200, 200)),
                    transforms.ToTensor(),
                ])

                if self.use_head_mask:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
                    head_region = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)
                    frame_black_background = np.zeros_like(rgb_frame)
                    head_region_with_black_background = cv2.add(head_region, frame_black_background)

                    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
                    head_crop = head_region_with_black_background[y:y + h, x:x + w]
                    head_crop = cv2.resize(head_crop, (420, 420))
                    cropped_images.append(head_crop)

                    head_tensor = transform(Image.fromarray(np.uint8(head_crop))).unsqueeze(0)
                    face_batches.append(head_tensor)
                else:
                    head_tensor = transform(Image.fromarray(np.uint8(rgb_frame))).unsqueeze(0)
                    face_batches.append(head_tensor)

                points_batch.append(points)
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                landmarks_batch.append(landmarks)

                self.landmarks_over_time.append(landmarks)

        return face_batches, points_batch, cropped_images, landmarks_batch

    def process_batch(self, frames):
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        all_face_batches = []
        all_points_batch = []
        all_cropped_images = []
        all_landmarks_batch = []

        for rgb_frame in rgb_frames:
            face_batches, points_batch, cropped_images, landmarks_batch = self.process_rgb_frame(rgb_frame)
            all_face_batches.extend(face_batches)
            all_points_batch.extend(points_batch)
            all_cropped_images.extend(cropped_images)
            all_landmarks_batch.extend(landmarks_batch)

        if all_face_batches:
            face_batch_tensor = torch.cat(all_face_batches).to(self.device)
            with torch.no_grad():
                face_batch_tensor = face_batch_tensor.unsqueeze(0)
                outputs = self.emotion_classifier(face_batch_tensor)
                outputs_mean = outputs.mean(dim=0) if outputs.dim() > 2 else outputs
                probabilities_batch = F.softmax(outputs_mean, dim=1)
                emotions_batch = torch.argmax(probabilities_batch, dim=1).cpu().numpy()
                print(emotions_batch)

                threshold = 0.5
                for idx, (emotion, points, rgb_frame) in enumerate(zip(emotions_batch, all_points_batch, rgb_frames)):
                    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
                    probabilities = probabilities_batch[idx].cpu().numpy()
                    max_prob = np.max(probabilities)
                    print(max_prob)
                    print(emotion)
                    emotion_text = self.get_emotion_label(emotion)
                    print(emotion_text)
                    cv2.putText(rgb_frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    self.update_trend_graph(emotions_batch)
                    self.update_pie_chart(probabilities)
                    self.update_bar_chart(probabilities)
                    self.draw_landmarks(rgb_frame, all_landmarks_batch[idx])
                    self.visualize_landmarks_over_time()

                    self.display_frame(rgb_frame)

        if self.use_head_mask:
            self.display_frames(all_cropped_images)
        else:
            self.display_frames(rgb_frames)

    def process_frame_batch(self, frame_batch):
        self.process_batch(frame_batch)

    def process_full_video(self, frames):
        total_frames = len(frames)
        
        # If the video has fewer frames than self.num_frames, process all available frames
        if total_frames <= self.num_frames:
            self.process_batch(frames)
        else:
            # Process the video in batches of self.num_frames
            for i in range(0, total_frames, self.num_frames):
                # Ensure not to exceed the total number of frames
                batch_frames = frames[i:min(i + self.num_frames, total_frames)]
                self.process_batch(batch_frames)

    
    def get_emotion_label(self, emotion_idx):
        emotions = ['Negative', 'Positive', 'Surprise']
        
        # Handle different types of emotion_idx
        if isinstance(emotion_idx, int):
            # Directly use the integer index
            return emotions[emotion_idx]
        elif isinstance(emotion_idx, list):
            # Extract the index from the list
            if emotion_idx:
                return emotions[emotion_idx[0]]
            else:
                return 'Unknown'  # Default or error case if the list is empty
        elif isinstance(emotion_idx, np.int64):
            # Handle numpy int64 type
            return emotions[int(emotion_idx)]
        else:
            return 'Unknown'  # Default or error case for unsupported types
        
    def display_frame(self, rgb_frame):
        frame = cv2.resize(rgb_frame, (self.video_frame.width(), self.video_frame.height()))
        frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(frame)
        self.video_frame.setPixmap(pixmap)

    def display_frames(self, rgb_frames):
        for rgb_frame in rgb_frames:
            self.display_frame(rgb_frame)

    def start_stop_webcam(self):
        if self.loaded_model_name is None:
            QMessageBox.warning(self, 'Warning', 'Please load a model first.')
            return

        if not self.is_webcam_running:           
            self.video_timer.stop()
            self.placeholder_label.hide()
            if not self.webcam.isOpened():
                if not self.webcam.open(0):
                    QMessageBox.warning(self, 'Warning', 'Failed to open webcam.')
                    return
            self.is_webcam_running = True
            self.frame_buffer = []
            self.webcam_timer.timeout.connect(self.update_webcam_frame)
            self.webcam_timer.start(1000 // self.video_fps)
            self.start_stop_button.setText('Stop Webcam')
        else:
            self.webcam.release()
            self.webcam_timer.stop()
            self.start_stop_button.setText('Start Webcam')
            self.is_webcam_running = False
            self.video_frame.clear()
            self.set_placeholder()

    def update_webcam_frame(self):
        if not self.is_webcam_running:
            return

        ret, frame = self.webcam.read()
        if ret:
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) >= self.num_frames:
                self.process_frame_batch(self.frame_buffer[:self.num_frames])
                self.frame_buffer = self.frame_buffer[self.stride:]

    def load_video(self):
        if self.loaded_model_name is None:
            QMessageBox.warning(self, 'Warning', 'Please load a model first.')
            return

        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.mp4 *.avi)')       
        self.webcam_timer.stop()
        self.video_timer.stop()
        self.placeholder_label.hide()
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.all_frames = []
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                self.all_frames.append(frame)
            self.total_frames = len(self.all_frames)
            self.video_slider.setMaximum(self.total_frames - 1)

            self.frame_buffer = []
            # self.process_full_video(self.all_frames)  # Process the entire video

            self.video_timer.timeout.connect(self.update_video_frame)
            self.video_timer.start(1000 // self.video_fps)

            self.play_pause_button.setText('Pause')
            self.is_video_playing = True

            self.current_frame_index = 0
            self.update_video_frame()

    def update_video_frame(self):
        if self.all_frames is None or not self.is_video_playing:
            return

        if self.current_frame_index < self.total_frames:
            frame = self.all_frames[self.current_frame_index]
            self.frame_buffer.append(frame)

            # Check if we have enough frames for a batch or if it's the final batch
            if len(self.frame_buffer) >= self.num_frames or self.current_frame_index + 1 == self.total_frames:
                # Process all available frames
                self.process_frame_batch(self.frame_buffer[:self.num_frames])
                self.frame_buffer = self.frame_buffer[self.stride:]

            self.video_slider.setValue(self.current_frame_index)
            self.current_frame_index += 1
        else:
            self.video_timer.stop()
            self.play_pause_button.setText('Play')
            self.is_video_playing = False
        
    def draw_landmarks(self, rgb_frame, landmarks):
        for index in self.landmark_indices:
            if index < len(landmarks):  # Ensure the index is within the bounds of the landmarks list
                landmark = landmarks[index]
                x, y, _ = landmark
                # Convert normalized coordinates to pixel coordinates
                x = int(x * rgb_frame.shape[1])
                y = int(y * rgb_frame.shape[0])
                # Draw the landmark as a small circle
                cv2.circle(rgb_frame, (x, y), 3, (0, 255, 0), -1)
    
    def visualize_landmarks_over_time(self):
        if len(self.landmarks_over_time) > 0 and self.landmark_plot_counter % self.landmark_plot_interval == 0:
            self.landmark_ax.clear()
            self.landmark_ax.set_title('Landmark Trends Over Time', fontsize=16, fontweight='bold')
            self.landmark_ax.set_xlabel('Frame Number', fontsize=12)
            self.landmark_ax.set_ylabel('Coordinate Value', fontsize=12)

            x_data = np.arange(len(self.landmarks_over_time))

            # Track min and max values across all landmarks
            all_y_data = []
            for idx in self.landmark_indices:
                if idx < len(self.landmarks_over_time[0]):
                    y_data = [self.landmarks_over_time[frame_idx][idx][0] for frame_idx in range(len(self.landmarks_over_time))]
                    all_y_data.extend(y_data)
                    self.landmark_ax.plot(x_data, y_data, label=idx, marker='o', linestyle='-', linewidth=2)
            
            # Set y-axis limits based on data range
            if all_y_data:
                self.landmark_ax.set_ylim([min(all_y_data) - 0.05, max(all_y_data) + 0.05])

            # Ensure legend is updated correctly
            self.landmark_ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
            self.landmark_ax.grid(True, linestyle='--', alpha=0.7)
            self.landmark_ax.set_facecolor('#f5f5f5')

            # Adjust subplot parameters to add space on the right side
            self.landmark_ax.figure.subplots_adjust(right=0.75, top=0.85, bottom=0.2, left=0.1)
            self.landmark_canvas.draw()

        self.landmark_plot_counter += 1

    def update_trend_graph(self, emotions):
        emotions_list = ['Negative', 'Positive', 'Surprise']
        if not hasattr(self, 'emotion_data'):
            self.emotion_data = {emo: [] for emo in emotions_list}
            self.time_data = []

        current_time = len(self.time_data)
        self.time_data.append(current_time)

        for emo in emotions_list:
            self.emotion_data[emo].append(list(emotions).count(emotions_list.index(emo)))

        self.trend_ax.clear()
        self.trend_ax.set_title('Emotion Trend Over Time', fontsize=16, fontweight='bold')
        self.trend_ax.set_xlabel('Time Step', fontsize=12)
        self.trend_ax.set_ylabel('Emotion Intensity', fontsize=12)

        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        for idx, emo in enumerate(emotions_list):
            self.trend_ax.plot(self.time_data, self.emotion_data[emo], label=emo, linestyle='-', marker='o', color=colors[idx])

        # Place legend outside to the right of the graph
        self.trend_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

        self.trend_ax.set_facecolor('#f5f5f5')
        self.trend_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        self.trend_ax.spines['top'].set_visible(False)
        self.trend_ax.spines['right'].set_visible(False)

        # Adjust layout to add space on the right side for the legend
        self.trend_ax.figure.subplots_adjust(right=0.75, top=0.85, bottom=0.1, left=0.1)
        self.trend_canvas.draw()


    def update_pie_chart(self, probabilities):
        emotions = ['Negative', 'Positive', 'Surprise']
        self.pie_ax.clear()
        self.pie_ax.set_title('Emotion Distribution', fontsize=16, fontweight='bold', pad=20)

        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        explode = (0, 0.1, 0)
        self.pie_ax.pie(probabilities, labels=emotions, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=True)
        self.pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Adjust layout to ensure everything fits well
        self.pie_ax.figure.subplots_adjust(top=0.75)
        self.pie_canvas.draw()

    def update_bar_chart(self, probabilities):
        emotions = ['Negative', 'Positive', 'Surprise']
        self.bar_ax.clear()
        bars = self.bar_ax.bar(emotions, probabilities, color=['#E74C3C', '#3498DB', '#2ECC71'], edgecolor='black')
        self.bar_ax.set_xlabel('Emotion Categories', fontsize=12)
        self.bar_ax.set_ylabel('Probability', fontsize=12)
        self.bar_ax.set_title('Emotion Analysis', fontsize=16, fontweight='bold')

        # Adding data labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            self.bar_ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', fontsize=10)

        self.bar_ax.set_facecolor('#f5f5f5')
        self.bar_ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        # Adjust layout to ensure everything fits well
        self.bar_ax.figure.subplots_adjust(left=0.20, right=0.80, top=0.85, bottom=0.2)
        self.bar_canvas.draw()
        
    def slider_pressed(self):
        if self.is_video_playing:
            self.video_timer.stop()

    def slider_released(self):
        if self.video_capture is not None:
            new_frame_index = self.video_slider.value()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_index)
            self.current_frame_index = new_frame_index
            if self.is_video_playing:
                self.video_timer.start(1000 // self.video_fps)

    @pyqtSlot(int)
    def update_frame_rate(self, value):
        if value == 0:
            return

        self.video_fps = value
        if self.is_webcam_running:
            self.webcam_timer.setInterval(1000 // self.video_fps)
        elif self.is_video_playing:
            self.video_timer.setInterval(1000 // self.video_fps)

        self.frame_rate_value_label.setText(f"{value}")

    def closeEvent(self, event):
        if self.video_capture is not None:
            self.video_capture.release()

        if self.is_webcam_running:
            self.webcam.release()

        for thread in self.threads:
            if thread.isRunning():
                thread.quit()
                thread.wait()

        event.accept()

def main():
    app = QApplication(sys.argv)

    # Load and set stylesheet
    with open('styles/styles.qss', 'r') as style_file:
        stylesheet = style_file.read()
    app.setStyleSheet(stylesheet)

    # Set application icon
    app.setWindowIcon(QIcon('icon.png'))

    # Run the login dialog
    login_dialog = LoginDialog()
    if login_dialog.exec_() == login_dialog.Accepted and login_dialog.accepted:
        # If login is accepted, show the main application window
        window = MicroExpressionApp()
        window.show()
        sys.exit(app.exec_())  # Start the application event loop
    else:
        sys.exit(0)  # Exit cleanly if login dialog is rejected

if __name__ == '__main__':
    main()
