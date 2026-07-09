import os
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog
import threading
import time
from analyzer import FaceGeometryAnalyzer
from predictor import BeautyPredictor

class CyberFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("CYBERFACE ANALYZER // HYBRID GEOMETRIC & NEURAL HUD")
        self.geometry("1100x800")
        self.resizable(False, False)

        # Set theme and color options
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Color palette
        self.bg_color = "#0a0d14"          # Deep space dark
        self.card_color = "#121824"        # Cyber card dark
        self.neon_cyan = "#00f0ff"         # Cyber cyan
        self.neon_green = "#00ff64"        # Cyber green
        self.neon_magenta = "#ff007f"      # Cyber magenta
        self.text_muted = "#8a99ad"        # Muted gray

        self.configure(fg_color=self.bg_color)

        # Initialize engines
        self.analyzer = FaceGeometryAnalyzer()
        self.predictor = BeautyPredictor()

        # Telegram Bot state
        self.bot_instance = None
        self.bot_running = False

        # Webcam & State variables
        self.cap = None
        self.webcam_running = False
        self.current_frame = None
        
        # Real-time state
        self.latest_face_crop = None
        self.latest_metrics = None
        self.last_predict_time = 0.0
        self.predict_thread_active = False
        self.latest_ai_score = None
        self.latest_raw_score = None

        # Photo tab state
        self.photo_webcam_active = False
        self.selected_pose = "Frontal"

        # Slots data (shared/unified for combined calculations)
        self.slots = {
            "Frontal": None,
            "Left Semi-profile": None,
            "Right Semi-profile": None,
            "Left Profile": None,
            "Right Profile": None
        }

        # Placeholder image for empty slots (prevents customtkinter ghosting bug)
        empty_pil = Image.new("RGB", (300, 200), color=self.card_color)
        self.empty_ctk_img = CTkImage(light_image=empty_pil, dark_image=empty_pil, size=(300, 200))

        # Build UI layout
        self.create_widgets()
        
        # Select initial slot in Photo tab to initialize visual highlights
        self.select_photo_slot("Frontal")

    def create_widgets(self):
        # ------------------- Left Column: CTkTabview -------------------
        self.tabview = ctk.CTkTabview(self, width=640, height=675, fg_color=self.bg_color, 
                                      segmented_button_fg_color="#101520",
                                      segmented_button_selected_color=self.neon_cyan,
                                      segmented_button_selected_hover_color="#00c8d6",
                                      segmented_button_unselected_color="#101520",
                                      text_color="#ffffff")
        self.tabview.place(x=20, y=5)
        
        self.tab_rt = self.tabview.add("Real-time Scanner")
        self.tab_photo = self.tabview.add("Photo Rating")
        
        # Setup Tab 1: Real-time Scan
        self.setup_realtime_tab()
        
        # Setup Tab 2: Photo Rating
        self.setup_photo_tab()

        # ------------------- Right Column: Dashboard -------------------
        self.dash_frame = ctk.CTkFrame(self, width=400, height=750, fg_color=self.card_color, border_width=1, border_color="#1f293d")
        self.dash_frame.place(x=680, y=20)

        # Header Title
        self.header_label = ctk.CTkLabel(self.dash_frame, text="CYBERFACE ANALYZER", 
                                         font=ctk.CTkFont(family="Consolas", size=18, weight="bold"), 
                                         text_color=self.neon_cyan)
        self.header_label.place(x=20, y=15)
        
        self.sub_label = ctk.CTkLabel(self.dash_frame, text="CYBER COMPUTE HUD 4.0", 
                                      font=ctk.CTkFont(family="Consolas", size=10), 
                                      text_color=self.text_muted)
        self.sub_label.place(x=20, y=40)

        # Target Group Config Selection
        self.group_title = ctk.CTkLabel(self.dash_frame, text="TARGET GROUP / CALIBRATION:", 
                                         font=ctk.CTkFont(family="Consolas", size=10, weight="bold"), 
                                         text_color=self.neon_cyan)
        self.group_title.place(x=20, y=70)

        self.group_combobox = ctk.CTkComboBox(self.dash_frame, 
                                               values=["Universal", "Young Man (14-20)", "Man (21+)", "Young Woman (14-20)", "Woman (21+)", "Strict Accuracy"],
                                               font=ctk.CTkFont(family="Consolas", size=12),
                                               width=360, height=30, command=self.on_group_changed)
        self.group_combobox.set("Universal")
        self.group_combobox.place(x=20, y=95)

        # HUD Overlay Toggle Switch
        self.hud_switch = ctk.CTkSwitch(self.dash_frame, text="SHOW HUD MESH OVERLAY", 
                                        font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
                                        text_color=self.neon_cyan,
                                        progress_color=self.neon_cyan, command=self.on_hud_toggle)
        self.hud_switch.select() # Default to enabled
        self.hud_switch.place(x=20, y=130)

        # Section 1: AI Score
        self.ai_section_label = ctk.CTkLabel(self.dash_frame, text="=== AI ATTRACTIVENESS SCORE ===", 
                                             font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                             text_color=self.neon_magenta)
        self.ai_section_label.place(x=20, y=155)

        self.score_label = ctk.CTkLabel(self.dash_frame, text="--.-- / 10.0", 
                                         font=ctk.CTkFont(family="Consolas", size=36, weight="bold"), 
                                         text_color=self.neon_magenta)
        self.score_label.place(x=20, y=175)

        self.tier_label = ctk.CTkLabel(self.dash_frame, text="", 
                                       font=ctk.CTkFont(family="Consolas", size=22, weight="bold"), 
                                       text_color=self.neon_cyan)
        self.tier_label.place(x=245, y=182)
        
        self.percentile_label = ctk.CTkLabel(self.dash_frame, text="", 
                                             font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                             text_color=self.neon_cyan)
        self.percentile_label.place(x=245, y=215)
        
        self.score_desc_label = ctk.CTkLabel(self.dash_frame, text="Neural Network Attractiveness Score", 
                                             font=ctk.CTkFont(family="Consolas", size=10), 
                                             text_color=self.text_muted)
        self.score_desc_label.place(x=20, y=235)

        self.potential_label = ctk.CTkLabel(self.dash_frame, text="POTENTIAL: --.--", 
                                             font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                             text_color=self.text_muted)
        self.potential_label.place(x=20, y=215)

        # Section 2: Geometric Metrics
        self.geom_section_label = ctk.CTkLabel(self.dash_frame, text="=== GEOMETRIC SYMMETRY HUD ===", 
                                               font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                               text_color=self.neon_green)
        self.geom_section_label.place(x=20, y=245)

        # Metric A: Symmetry
        self.lbl_sym = ctk.CTkLabel(self.dash_frame, text="Facial Symmetry: --%", 
                                    font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), 
                                    text_color="#ffffff")
        self.lbl_sym.place(x=20, y=270)
        self.pb_sym = ctk.CTkProgressBar(self.dash_frame, width=360, height=10, 
                                         fg_color="#101a24", progress_color=self.neon_green)
        self.pb_sym.set(0)
        self.pb_sym.place(x=20, y=295)

        # Metric B: Golden Ratio
        self.lbl_gr = ctk.CTkLabel(self.dash_frame, text="Golden Ratio (1.618): --%", 
                                   font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), 
                                   text_color="#ffffff")
        self.lbl_gr.place(x=20, y=315)
        self.pb_gr = ctk.CTkProgressBar(self.dash_frame, width=360, height=10, 
                                        fg_color="#101a24", progress_color=self.neon_cyan)
        self.pb_gr.set(0)
        self.pb_gr.place(x=20, y=340)

        # Metric C: Overall Geometry
        self.lbl_geom = ctk.CTkLabel(self.dash_frame, text="Overall Geometry Index: --%", 
                                     font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), 
                                     text_color="#ffffff")
        self.lbl_geom.place(x=20, y=360)
        self.pb_geom = ctk.CTkProgressBar(self.dash_frame, width=360, height=10, 
                                          fg_color="#101a24", progress_color="#8a5cf6")
        self.pb_geom.set(0)
        self.pb_geom.place(x=20, y=385)

        # Section 3: Telemetry Logs
        self.log_section_label = ctk.CTkLabel(self.dash_frame, text="=== TELEMETRY ANALYSIS LOG ===", 
                                              font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                              text_color=self.neon_cyan)
        self.log_section_label.place(x=20, y=425)

        self.log_box = ctk.CTkTextbox(self.dash_frame, width=360, height=180, 
                                      fg_color="#080c14", text_color=self.neon_cyan, 
                                      font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.place(x=20, y=450)
        self.log_box.insert("0.0", "Waiting for face detection...\n")
        self.log_box.configure(state="disabled")

        # Section 4: Telegram Bot Integration
        self.bot_section_label = ctk.CTkLabel(self.dash_frame, text="=== TELEGRAM BOT INTEGRATION ===", 
                                              font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                              text_color=self.neon_magenta)
        self.bot_section_label.place(x=20, y=640)

        self.bot_token_entry = ctk.CTkEntry(self.dash_frame, width=250, height=30, 
                                            placeholder_text="Enter Telegram Bot Token...", 
                                            show="*", font=ctk.CTkFont(family="Consolas", size=11))
        self.bot_token_entry.place(x=20, y=665)
        self.bind_entry_shortcuts(self.bot_token_entry)

        self.btn_toggle_bot = ctk.CTkButton(self.dash_frame, text="START BOT", 
                                            fg_color="#1f2d3d", hover_color=self.neon_green, 
                                            border_width=1, border_color=self.neon_green,
                                            text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                                            width=100, height=30, command=self.toggle_telegram_bot)
        self.btn_toggle_bot.place(x=280, y=665)

        self.lbl_bot_status = ctk.CTkLabel(self.dash_frame, text="BOT STATUS: INACTIVE", 
                                           font=ctk.CTkFont(family="Consolas", size=10, weight="bold"), 
                                           text_color=self.text_muted)
        self.lbl_bot_status.place(x=20, y=705)

    # ------------------- Tab 1: Real-time Scan Layout -------------------
    def setup_realtime_tab(self):
        # Video Frame inside tab
        self.video_frame = ctk.CTkFrame(self.tab_rt, width=600, height=400, fg_color=self.card_color, border_width=1, border_color=self.neon_cyan)
        self.video_frame.place(x=10, y=10)

        self.video_label = ctk.CTkLabel(self.video_frame, text="CAMERA CLOSED\n\nStart webcam to begin scanning", 
                                         font=ctk.CTkFont(family="Consolas", size=13, weight="bold"), 
                                         text_color=self.text_muted)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Pose HUD label
        self.lbl_hud_pose = ctk.CTkLabel(self.video_frame, text="POSE: WAITING", 
                                         fg_color="#080c14", text_color=self.neon_cyan, 
                                         font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), 
                                         corner_radius=4, width=220, height=28)
        self.lbl_hud_pose.place(x=15, y=15)

        # Real-time control frame
        self.rt_controls_frame = ctk.CTkFrame(self.tab_rt, width=600, height=195, fg_color=self.card_color, border_width=1, border_color="#1f293d")
        self.rt_controls_frame.place(x=10, y=420)

        # Row 1: Actions
        self.btn_webcam = ctk.CTkButton(self.rt_controls_frame, text="START CAMERA", 
                                         fg_color="#1f2d3d", hover_color=self.neon_cyan, 
                                         border_width=1, border_color=self.neon_cyan,
                                         text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                                         width=160, height=35, command=self.toggle_webcam)
        self.btn_webcam.place(x=15, y=15)

        self.btn_capture = ctk.CTkButton(self.rt_controls_frame, text="LOCK POSE", 
                                         fg_color="#1f2d3d", hover_color=self.neon_magenta, 
                                         border_width=1, border_color=self.neon_magenta,
                                         text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                                         state="disabled", width=180, height=35, command=self.auto_capture_slot)
        self.btn_capture.place(x=185, y=15)

        self.btn_reset_rt = ctk.CTkButton(self.rt_controls_frame, text="RESET", 
                                           fg_color="#1a1e29", hover_color="#ff3333",
                                           text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=11),
                                           width=80, height=35, command=self.reset_slots)
        self.btn_reset_rt.place(x=375, y=15)

        self.btn_combined_rt = ctk.CTkButton(self.rt_controls_frame, text="COMBINED SCORE", 
                                              fg_color="#251a3d", hover_color="#8a5cf6", 
                                              border_width=1, border_color="#8a5cf6",
                                              text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                                              state="disabled", width=120, height=35, command=self.calculate_combined_rating)
        self.btn_combined_rt.place(x=465, y=15)

        # Row 2: Slots Buttons
        slot_font = ctk.CTkFont(family="Consolas", size=10)
        self.btn_rt_frontal = ctk.CTkButton(self.rt_controls_frame, text="⬡ Frontal", 
                                             fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                             font=slot_font, width=105, height=28, command=lambda: self.manual_capture_slot("Frontal"))
        self.btn_rt_frontal.place(x=15, y=65)

        self.btn_rt_lsemi = ctk.CTkButton(self.rt_controls_frame, text="⬡ L.Semi", 
                                           fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                           font=slot_font, width=105, height=28, command=lambda: self.manual_capture_slot("Left Semi-profile"))
        self.btn_rt_lsemi.place(x=130, y=65)

        self.btn_rt_rsemi = ctk.CTkButton(self.rt_controls_frame, text="⬡ R.Semi", 
                                           fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                           font=slot_font, width=105, height=28, command=lambda: self.manual_capture_slot("Right Semi-profile"))
        self.btn_rt_rsemi.place(x=245, y=65)

        self.btn_rt_lprof = ctk.CTkButton(self.rt_controls_frame, text="⬡ L.Profile", 
                                           fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                           font=slot_font, width=105, height=28, command=lambda: self.manual_capture_slot("Left Profile"))
        self.btn_rt_lprof.place(x=360, y=65)

        self.btn_rt_rprof = ctk.CTkButton(self.rt_controls_frame, text="⬡ R.Profile", 
                                           fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                           font=slot_font, width=105, height=28, command=lambda: self.manual_capture_slot("Right Profile"))
        self.btn_rt_rprof.place(x=475, y=65)

        # Row 3: Status Labels
        self.status_label_rt = ctk.CTkLabel(self.rt_controls_frame, text="STATUS: WAITING FOR START", 
                                            font=ctk.CTkFont(family="Consolas", size=11), text_color=self.neon_cyan)
        self.status_label_rt.place(x=15, y=110)

        self.info_label_rt = ctk.CTkLabel(self.rt_controls_frame, text="In scanner mode AI outputs real-time rating score", 
                                          font=ctk.CTkFont(family="Consolas", size=9), text_color=self.text_muted)
        self.info_label_rt.place(x=15, y=135)

    # ------------------- Tab 2: Photo Rating Layout -------------------
    def setup_photo_tab(self):
        # Left Side: Slot preview and upload list (width 300)
        self.photo_preview_frame = ctk.CTkFrame(self.tab_photo, width=300, height=200, fg_color=self.card_color, border_width=1, border_color=self.neon_cyan)
        self.photo_preview_frame.place(x=10, y=10)

        self.photo_preview_label = ctk.CTkLabel(self.photo_preview_frame, text="NO PHOTO\n\nUpload photo for selected slot", 
                                                font=ctk.CTkFont(family="Consolas", size=11), text_color=self.text_muted)
        self.photo_preview_label.place(relx=0.5, rely=0.5, anchor="center")

        # Row: Web-camera actions inside Photo tab
        action_font = ctk.CTkFont(family="Consolas", size=11, weight="bold")
        self.btn_photo_webcam = ctk.CTkButton(self.tab_photo, text="START CAMERA", 
                                              fg_color="#1f2d3d", hover_color=self.neon_cyan, border_width=1, border_color=self.neon_cyan,
                                              text_color="#ffffff", font=action_font, width=90, height=30, command=self.toggle_photo_webcam)
        self.btn_photo_webcam.place(x=10, y=215)

        self.btn_photo_capture = ctk.CTkButton(self.tab_photo, text="TAKE SNAPSHOT", 
                                               fg_color="#1f2d3d", hover_color=self.neon_magenta, border_width=1, border_color=self.neon_magenta,
                                               text_color="#ffffff", font=action_font, state="disabled", width=95, height=30, command=self.capture_photo_to_slot)
        self.btn_photo_capture.place(x=105, y=215)

        self.btn_photo_file = ctk.CTkButton(self.tab_photo, text="UPLOAD FILE", 
                                            fg_color="#1f2d3d", hover_color=self.neon_green, border_width=1, border_color=self.neon_green,
                                            text_color="#ffffff", font=action_font, width=90, height=30, command=self.upload_photo_file_click)
        self.btn_photo_file.place(x=205, y=215)

        # Selected Slot Title
        self.lbl_selected_slot = ctk.CTkLabel(self.tab_photo, text="SELECTED SLOT: FRONTAL", 
                                              font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color=self.neon_cyan)
        self.lbl_selected_slot.place(x=10, y=255)

        # 5 photo slots buttons (now act as selection list)
        btn_h = 32
        font_p = ctk.CTkFont(family="Consolas", size=11)
        
        self.btn_ph_frontal = ctk.CTkButton(self.tab_photo, text="⬡ Slot 1: Frontal", 
                                            fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                            font=font_p, width=280, height=btn_h, command=lambda: self.select_photo_slot("Frontal"))
        self.btn_ph_frontal.place(x=10, y=285)

        self.btn_ph_lsemi = ctk.CTkButton(self.tab_photo, text="⬡ Slot 2: L. Semi-profile", 
                                          fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                          font=font_p, width=280, height=btn_h, command=lambda: self.select_photo_slot("Left Semi-profile"))
        self.btn_ph_lsemi.place(x=10, y=320)

        self.btn_ph_rsemi = ctk.CTkButton(self.tab_photo, text="⬡ Slot 3: R. Semi-profile", 
                                          fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                          font=font_p, width=280, height=btn_h, command=lambda: self.select_photo_slot("Right Semi-profile"))
        self.btn_ph_rsemi.place(x=10, y=355)

        self.btn_ph_lprof = ctk.CTkButton(self.tab_photo, text="⬡ Slot 4: L. Profile", 
                                          fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                          font=font_p, width=280, height=btn_h, command=lambda: self.select_photo_slot("Left Profile"))
        self.btn_ph_lprof.place(x=10, y=390)

        self.btn_ph_rprof = ctk.CTkButton(self.tab_photo, text="⬡ Slot 5: R. Profile", 
                                          fg_color="#101520", hover_color="#202a3d", text_color=self.text_muted, 
                                          font=font_p, width=280, height=btn_h, command=lambda: self.select_photo_slot("Right Profile"))
        self.btn_ph_rprof.place(x=10, y=425)

        # Action Buttons
        self.btn_reset_photo = ctk.CTkButton(self.tab_photo, text="RESET PHOTOS", 
                                             fg_color="#1a1e29", hover_color="#ff3333", text_color="#ffffff",
                                             font=font_p, width=130, height=35, command=self.reset_slots)
        self.btn_reset_photo.place(x=10, y=465)

        self.btn_run_deep = ctk.CTkButton(self.tab_photo, text="DEEP ANALYSIS", 
                                          fg_color="#251a3d", hover_color=self.neon_magenta, border_width=1, border_color=self.neon_magenta,
                                          text_color="#ffffff", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                                          state="disabled", width=140, height=35, command=self.start_deep_analysis_thread)
        self.btn_run_deep.place(x=150, y=465)

        # Progress visualizer
        self.deep_pb = ctk.CTkProgressBar(self.tab_photo, width=280, height=12, fg_color="#101a24", progress_color=self.neon_magenta)
        self.deep_pb.set(0)
        self.deep_pb.place(x=10, y=510)

        self.deep_status_lbl = ctk.CTkLabel(self.tab_photo, text="Waiting for photo uploads...", 
                                            font=ctk.CTkFont(family="Consolas", size=10), text_color=self.text_muted)
        self.deep_status_lbl.place(x=10, y=532)

        # Right Side: Deep analysis visual console log (width 290)
        self.console_title = ctk.CTkLabel(self.tab_photo, text="=== AI DEEP DIAGNOSTICS ===", 
                                          font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color=self.neon_magenta)
        self.console_title.place(x=320, y=10)

        self.deep_log_box = ctk.CTkTextbox(self.tab_photo, width=290, height=510, 
                                            fg_color="#080c14", text_color=self.neon_magenta, 
                                            font=ctk.CTkFont(family="Consolas", size=11))
        self.deep_log_box.place(x=320, y=35)
        self.deep_log_box.insert("0.0", "AI console is ready.\n\nFill slots via camera or upload files (Frontal + any profile) and run 'DEEP ANALYSIS'.\n")
        self.deep_log_box.configure(state="disabled")

    # ------------------- Select Slot in Photo Tab -------------------
    def select_photo_slot(self, pose):
        self.selected_pose = pose
        self.lbl_selected_slot.configure(text=f"SELECTED SLOT: {pose.upper()}")
        
        # Reset borders
        buttons = {
            "Frontal": self.btn_ph_frontal,
            "Left Semi-profile": self.btn_ph_lsemi,
            "Right Semi-profile": self.btn_ph_rsemi,
            "Left Profile": self.btn_ph_lprof,
            "Right Profile": self.btn_ph_rprof
        }
        for k, btn in buttons.items():
            if k == pose:
                btn.configure(border_width=1, border_color=self.neon_cyan)
            else:
                btn.configure(border_width=0)

        # Display preview if slot has image data and webcam is not running
        if not self.photo_webcam_active:
            slot_data = self.slots[pose]
            if slot_data and slot_data["frame"] is not None:
                group = self.get_selected_target_group_english()
                hud_frame, _, _ = self.analyzer.analyze_frame(slot_data["frame"], target_group=group, draw_hud=self.hud_switch.get())
                self.display_photo_preview(hud_frame)
            else:
                self.photo_preview_label.configure(image=self.empty_ctk_img, text=f"NO PHOTO\n\nUpload or capture frame\nfor slot: {pose}")

    # ------------------- Actions & Computations -------------------
    def get_selected_target_group_english(self):
        val = self.group_combobox.get()
        mapping = {
            "Universal": "Universal",
            "Young Man (14-20)": "Young Man",
            "Man (21+)": "Man",
            "Young Woman (14-20)": "Young Woman",
            "Woman (21+)": "Woman",
            "Strict Accuracy": "Strict"
        }
        return mapping.get(val, "Universal")

    def on_group_changed(self, event=None):
        self.status_label_rt.configure(text=f"CALIBRATION CHANGED: {self.group_combobox.get()}")
        if self.current_frame is not None and self.webcam_running:
            self.process_and_display_frame(self.current_frame)

    def on_hud_toggle(self):
        if self.current_frame is not None and self.webcam_running:
            self.process_and_display_frame(self.current_frame)
        elif not self.photo_webcam_active:
            slot_data = self.slots[self.selected_pose]
            if slot_data and slot_data["frame"] is not None:
                group = self.get_selected_target_group_english()
                hud_frame, _, _ = self.analyzer.analyze_frame(slot_data["frame"], target_group=group, draw_hud=self.hud_switch.get())
                self.display_photo_preview(hud_frame)

    def toggle_photo_webcam(self):
        if self.photo_webcam_active:
            # Stop webcam in Photo Tab
            self.photo_webcam_active = False
            self.webcam_running = False
            self.btn_photo_webcam.configure(text="START CAMERA", fg_color="#1f2d3d", border_color=self.neon_cyan)
            self.btn_photo_capture.configure(state="disabled")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.select_photo_slot(self.selected_pose) # Restore preview
        else:
            # Stop webcam in Real-time tab if running
            if self.webcam_running:
                self.toggle_webcam()
                
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.deep_status_lbl.configure(text="Webcam connection error.")
                self.cap = None
                return
                
            self.photo_webcam_active = True
            self.webcam_running = True
            self.btn_photo_webcam.configure(text="STOP CAMERA", fg_color=self.neon_magenta, border_color=self.neon_magenta)
            self.btn_photo_capture.configure(state="normal")
            self.deep_status_lbl.configure(text="Camera active. Select a face pose.")
            self.update_cam()

    def capture_photo_to_slot(self):
        if not self.photo_webcam_active or self.current_frame is None:
            return
            
        if self.latest_metrics is None or not self.latest_metrics["detected"]:
            # Fallback to manual capture
            face_crop, metrics = self.get_fallback_metrics_and_crop(self.current_frame, self.selected_pose)
            group = self.get_selected_target_group_english()
            geom_data = {
                "symmetry": metrics.get("symmetry", 0.0),
                "golden_ratio": metrics.get("golden_ratio", 0.0),
                "overall_geom": metrics.get("overall_geom", 0.0)
            }
            score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=True, geom_data=geom_data)
            metrics["ai_score"] = score_10
            metrics["raw_score"] = raw_score
            self.save_to_slot(self.selected_pose, self.current_frame, face_crop, metrics, is_webcam=True)
            self.toggle_photo_webcam()
            self.deep_status_lbl.configure(text=f"Snapshot saved to slot {self.selected_pose} (no landmarks)!")
            return

        pose = self.selected_pose
        detected_pose = self.latest_metrics["pose"]
        
        # Save snapshot
        self.save_to_slot(pose, self.current_frame, self.latest_face_crop, self.latest_metrics, is_webcam=True)
        
        # Stop webcam after snapshot
        self.toggle_photo_webcam()
        
        if detected_pose != pose:
            self.deep_status_lbl.configure(text=f"Snapshot saved. Warning: detected pose {detected_pose}!")
        else:
            self.deep_status_lbl.configure(text=f"Snapshot successfully saved to slot {pose}!")

    def upload_photo_file_click(self):
        pose = self.selected_pose
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return
            
        # Unicode-safe image loading for Windows (handles Russian/Cyrillic paths)
        try:
            pil_img = Image.open(file_path).convert("RGB")
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[GUI] Error loading image: {e}")
            frame = None

        if frame is None:
            self.deep_status_lbl.configure(text="Failed to load image.")
            return

        # Resize and pad border to 640x480
        h, w = frame.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        
        padded_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dx = (640 - new_w) // 2
        dy = (480 - new_h) // 2
        padded_frame[dy:dy+new_h, dx:dx+new_w] = frame

        group = self.get_selected_target_group_english()

        # Analyze
        hud_frame, face_crop, metrics = self.analyzer.analyze_frame(padded_frame, target_group=group, draw_hud=self.hud_switch.get())
        
        if not metrics["detected"]:
            # Fallback
            face_crop, metrics = self.get_fallback_metrics_and_crop(padded_frame, pose)
            geom_data = {
                "symmetry": metrics.get("symmetry", 0.0),
                "golden_ratio": metrics.get("golden_ratio", 0.0),
                "overall_geom": metrics.get("overall_geom", 0.0)
            }
            score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=False, geom_data=geom_data)
            metrics["ai_score"] = score_10
            metrics["raw_score"] = raw_score
            self.deep_status_lbl.configure(text=f"Photo loaded to slot: {pose} (no landmarks)!")
            self.save_to_slot(pose, padded_frame, face_crop, metrics, is_webcam=False)
            return

        # Quick prediction score to initialize slot
        geom_data = {
            "symmetry": metrics.get("symmetry", 0.0),
            "golden_ratio": metrics.get("golden_ratio", 0.0),
            "overall_geom": metrics.get("overall_geom", 0.0)
        }
        score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=False, geom_data=geom_data)
        if score_10 is not None:
            metrics["ai_score"] = score_10
            metrics["raw_score"] = raw_score

        # Check for profile mismatches
        detected_pose = metrics["pose"]

        if detected_pose != pose:
            self.deep_status_lbl.configure(text=f"[Warning] You loaded {detected_pose} into {pose} slot!")
        else:
            self.deep_status_lbl.configure(text=f"Photo successfully loaded to slot {pose}!")
            
        self.save_to_slot(pose, padded_frame, face_crop, metrics, is_webcam=False)

    def toggle_webcam(self):
        if self.webcam_running:
            self.webcam_running = False
            self.btn_webcam.configure(text="START CAMERA", fg_color="#1f2d3d", border_color=self.neon_cyan)
            self.btn_capture.configure(state="disabled")
            self.status_label_rt.configure(text="STATUS: CAMERA CLOSED", text_color=self.neon_cyan)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.video_label.configure(image=None, text="CAMERA CLOSED\n\nStart webcam to begin scanning")
            self.lbl_hud_pose.configure(text="POSE: CLOSED", text_color=self.text_muted)
        else:
            # Stop photo webcam first if running
            if self.photo_webcam_active:
                self.toggle_photo_webcam()
                
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label_rt.configure(text="STATUS: CAMERA ERROR", text_color=self.neon_magenta)
                self.cap = None
                return
                
            self.webcam_running = True
            self.btn_webcam.configure(text="STOP CAMERA", fg_color=self.neon_magenta, border_color=self.neon_magenta)
            self.btn_capture.configure(state="normal")
            self.status_label_rt.configure(text="STATUS: ACTIVE SCANNING", text_color=self.neon_green)
            self.update_cam()

    def update_cam(self):
        if not self.webcam_running:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.status_label_rt.configure(text="STATUS: FRAME CAPTURE ERROR", text_color=self.neon_magenta)
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        self.current_frame = frame.copy()

        self.process_and_display_frame(frame)
        self.after(15, self.update_cam)

    def process_and_display_frame(self, frame):
        group = self.get_selected_target_group_english()
        
        # Analyze frame
        hud_frame, face_crop, metrics = self.analyzer.analyze_frame(frame, target_group=group, draw_hud=self.hud_switch.get())
        
        if metrics["detected"] and face_crop is not None:
            # Asynchronous quick prediction to prevent webcam frame lag
            now = time.time()
            if not self.predict_thread_active and (now - self.last_predict_time > 0.4):
                self.predict_thread_active = True
                geom_data = {
                    "symmetry": metrics.get("symmetry", 0),
                    "golden_ratio": metrics.get("golden_ratio", 0),
                    "overall_geom": metrics.get("overall_geom", 0)
                }
                threading.Thread(
                    target=self.bg_predict_worker, 
                    args=(face_crop.copy(), group, geom_data), 
                    daemon=True
                ).start()
            
            # Inject latest computed score
            if self.latest_ai_score is not None:
                metrics["ai_score"] = self.latest_ai_score
                metrics["raw_score"] = self.latest_raw_score
        
        self.latest_face_crop = face_crop
        self.latest_metrics = metrics

        # Update HUD labels
        if metrics["detected"]:
            pose_map = {
                "Frontal": ("POSE: FRONTAL", self.neon_cyan),
                "Left Semi-profile": ("POSE: L. SEMI-PROFILE", self.neon_magenta),
                "Right Semi-profile": ("POSE: R. SEMI-PROFILE", self.neon_magenta),
                "Left Profile": ("POSE: L. PROFILE", self.neon_green),
                "Right Profile": ("POSE: R. PROFILE", self.neon_green)
            }
            text, color = pose_map.get(metrics["pose"], ("POSE: UNKNOWN", self.text_muted))
            self.lbl_hud_pose.configure(text=f"{text} ({metrics['yaw']:.1f}°)", text_color=color)
        else:
            self.lbl_hud_pose.configure(text="POSE: WAITING", text_color=self.text_muted)

        self.update_ui_with_metrics(metrics)
        
        # Display image based on active webcam mode
        if self.photo_webcam_active:
            frame_resized = cv2.resize(hud_frame, (300, 200))
            self.display_photo_preview(frame_resized)
        else:
            frame_resized = cv2.resize(hud_frame, (600, 400))
            self.display_image(frame_resized)

    def bg_predict_worker(self, crop, group, geom_data=None):
        try:
            score_10, raw = self.predictor.predict(crop, target_group=group, is_webcam=True, geom_data=geom_data)
            self.latest_ai_score = score_10
            self.latest_raw_score = raw
        except Exception as e:
            print(f"[GUI] Error in bg_predict_worker: {e}")
        finally:
            self.predict_thread_active = False
            self.last_predict_time = time.time()

    def display_photo_preview(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        ctk_img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(300, 200))
        self.photo_preview_label.configure(image=ctk_img, text="")
        self.photo_preview_label.image = ctk_img

    def get_fallback_metrics_and_crop(self, frame, pose):
        h, w = frame.shape[:2]
        # Crop center 60% of the image
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        face_crop = frame[y1:y2, x1:x2].copy()
        
        metrics = {
            "detected": True,
            "yaw": 60.0 if "Left" in pose else -60.0,
            "pose": pose,
            "symmetry": 0.0,
            "golden_ratio": 70.0,
            "overall_geom": 70.0,
            "details": ["Profile manually fixed.", "Media-markup unavailable (side profile).", "Applied standard contour estimation."]
        }
        return face_crop, metrics

    def auto_capture_slot(self):
        if self.latest_metrics is None or not self.latest_metrics["detected"]:
            self.status_label_rt.configure(text="ERROR: FACE NOT DETECTED", text_color=self.neon_magenta)
            return
        pose = self.latest_metrics["pose"]
        self.save_to_slot(pose, self.current_frame, self.latest_face_crop, self.latest_metrics)

    def manual_capture_slot(self, pose):
        if self.current_frame is None:
            self.status_label_rt.configure(text="ERROR: NO ACTIVE FRAME", text_color=self.neon_magenta)
            return
        if self.latest_metrics is None or not self.latest_metrics["detected"]:
            # Fallback
            face_crop, metrics = self.get_fallback_metrics_and_crop(self.current_frame, pose)
            group = self.get_selected_target_group_english()
            geom_data = {
                "symmetry": metrics.get("symmetry", 0.0),
                "golden_ratio": metrics.get("golden_ratio", 0.0),
                "overall_geom": metrics.get("overall_geom", 0.0)
            }
            score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=True, geom_data=geom_data)
            metrics["ai_score"] = score_10
            metrics["raw_score"] = raw_score
            self.save_to_slot(pose, self.current_frame, face_crop, metrics, is_webcam=True)
            self.status_label_rt.configure(text=f"SAVED: {pose.upper()} (MANUAL)", text_color=self.neon_cyan)
            return
        self.save_to_slot(pose, self.current_frame, self.latest_face_crop, self.latest_metrics, is_webcam=True)

    def save_to_slot(self, pose, frame, face_crop, metrics, is_webcam=False):
        if metrics.get("ai_score") is None or metrics.get("ai_score") == 0.0:
            if face_crop is not None:
                group = self.get_selected_target_group_english()
                geom_data = {
                    "symmetry": metrics.get("symmetry", 0.0),
                    "golden_ratio": metrics.get("golden_ratio", 0.0),
                    "overall_geom": metrics.get("overall_geom", 0.0)
                }
                score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=is_webcam, geom_data=geom_data)
                metrics["ai_score"] = score_10
                metrics["raw_score"] = raw_score

        self.slots[pose] = {
            "score": metrics.get("ai_score", 0.0),
            "raw_score": metrics.get("raw_score", 0.0),
            "geom_score": metrics.get("overall_geom", 0.0),
            "symmetry": metrics.get("symmetry", 0.0),
            "golden_ratio": metrics.get("golden_ratio", 0.0),
            "details": list(metrics.get("details", [])),
            "frame": frame.copy(),
            "face_crop": face_crop.copy() if face_crop is not None else None,
            "is_webcam": is_webcam
        }

        # Update button texts & colors
        score = self.slots[pose]["score"]
        self.update_button_states_for_slot(pose, score)
        
        self.status_label_rt.configure(text=f"SAVED: {pose.upper()} ({score:.2f})", text_color=self.neon_green)
        
        # Check if combined analysis should be enabled
        if self.slots["Frontal"] is not None and any(self.slots[k] is not None for k in self.slots if k != "Frontal"):
            self.btn_combined_rt.configure(state="normal")
            self.btn_run_deep.configure(state="normal")

    def update_button_states_for_slot(self, pose, score):
        tier_text, tier_color = self.get_tier_info(score)
        score_str = f"{score:.2f}" if score is not None else "0.00"
        
        buttons_rt = {
            "Frontal": (self.btn_rt_frontal, f"⬢ Frontal ({score_str})"),
            "Left Semi-profile": (self.btn_rt_lsemi, f"⬢ L.Semi ({score_str})"),
            "Right Semi-profile": (self.btn_rt_rsemi, f"⬢ R.Semi ({score_str})"),
            "Left Profile": (self.btn_rt_lprof, f"⬢ L.Profile ({score_str})"),
            "Right Profile": (self.btn_rt_rprof, f"⬢ R.Profile ({score_str})")
        }
        buttons_ph = {
            "Frontal": (self.btn_ph_frontal, f"⬢ Slot 1: Frontal ({score_str})"),
            "Left Semi-profile": (self.btn_ph_lsemi, f"⬢ Slot 2: L. Semi-profile ({score_str})"),
            "Right Semi-profile": (self.btn_ph_rsemi, f"⬢ Slot 3: R. Semi-profile ({score_str})"),
            "Left Profile": (self.btn_ph_lprof, f"⬢ Slot 4: L. Profile ({score_str})"),
            "Right Profile": (self.btn_ph_rprof, f"⬢ Slot 5: R. Profile ({score_str})")
        }

        # Update RT
        btn_rt, text_rt = buttons_rt[pose]
        btn_rt.configure(text=text_rt, fg_color=self.card_color, text_color=tier_color, border_width=1, border_color=tier_color)
        
        # Update Photo
        btn_ph, text_ph = buttons_ph[pose]
        is_selected = (self.selected_pose == pose)
        border_col = self.neon_cyan if is_selected else tier_color
        btn_ph.configure(text=text_ph, fg_color=self.card_color, text_color=tier_color, border_width=1, border_color=border_col)

        # Refresh preview if active selection matches pose
        if self.selected_pose == pose and not self.photo_webcam_active:
            slot_data = self.slots[pose]
            if slot_data and slot_data["frame"] is not None:
                group = self.get_selected_target_group_english()
                hud_frame, _, _ = self.analyzer.analyze_frame(slot_data["frame"], target_group=group, draw_hud=self.hud_switch.get())
                self.display_photo_preview(hud_frame)

    # ------------------- Deep Multi-Angle Analysis -------------------
    def start_deep_analysis_thread(self):
        self.btn_run_deep.configure(state="disabled")
        self.btn_reset_photo.configure(state="disabled")
        self.btn_combined_rt.configure(state="disabled")
        
        self.deep_log_box.configure(state="normal")
        self.deep_log_box.delete("0.0", "end")
        self.deep_log_box.insert("end", "=== INITIALIZING AI DEEP ANALYSIS ===\n")
        self.deep_log_box.insert("end", f"Calibration: {self.group_combobox.get()}\n")
        self.deep_log_box.configure(state="disabled")
        
        threading.Thread(target=self.run_deep_analysis_compute, daemon=True).start()

    def run_deep_analysis_compute(self):
        group = self.get_selected_target_group_english()
        filled_slots = {k: v for k, v in self.slots.items() if v is not None}
        total_steps = len(filled_slots)
        
        self.log_to_console("[SYSTEM] Found active slots: " + str(total_steps))
        
        for idx, (pose, data) in enumerate(filled_slots.items()):
            self.log_to_console(f"\n[POSE] Starting deep analysis: {pose.upper()}")
            
            def progress_cb(pct, status_text):
                overall_progress = (idx / total_steps) + (pct / total_steps)
                self.deep_pb.set(overall_progress)
                self.deep_status_lbl.configure(text=f"Analyzing {pose}: {status_text}")
                self.log_to_console(f"  * {status_text} (TTA {int(pct*100)}%)")
                self.update()

            face_img = data["face_crop"]
            is_webcam = data.get("is_webcam", False)
            # Run Deep TTA prediction
            score_10, raw_score = self.predictor.predict_deep(face_img, target_group=group, is_webcam=is_webcam, progress_callback=progress_cb)
            
            if score_10 is not None:
                self.slots[pose]["score"] = score_10
                self.slots[pose]["raw_score"] = raw_score
                self.log_to_console(f"  [RESULT] TTA pose score: {score_10:.2f}/10.0 (Raw: {raw_score:.2f})")
                
                # Update GUI buttons text on main thread
                self.after(10, lambda p=pose, s=score_10: self.update_button_states_for_slot(p, s))
                
        self.deep_pb.set(1.0)
        self.deep_status_lbl.configure(text="Consolidating combined AI ratings...")
        self.log_to_console("\n[SYSTEM] All slots processed. Calculating combined ratings...")
        
        self.after(200, self.calculate_combined_rating)
        
        self.btn_run_deep.configure(state="normal")
        self.btn_reset_photo.configure(state="normal")

    def log_to_console(self, text):
        self.deep_log_box.configure(state="normal")
        self.deep_log_box.insert("end", text + "\n")
        self.deep_log_box.see("end")
        self.deep_log_box.configure(state="disabled")

    # ------------------- Shared Combined Calculation -------------------
    def calculate_combined_rating(self):
        frontal = self.slots["Frontal"]
        lsemi = self.slots["Left Semi-profile"]
        rsemi = self.slots["Right Semi-profile"]
        lprof = self.slots["Left Profile"]
        rprof = self.slots["Right Profile"]

        if frontal is None:
            self.status_label_rt.configure(text="ERROR: FRONTAL SLOT REQUIRED FOR COMBINED CALCULATION", text_color=self.neon_magenta)
            self.deep_status_lbl.configure(text="Error: Frontal slot is empty.")
            return

        active_slots = {}
        for k in self.slots:
            if self.slots[k] is not None and self.slots[k].get("score") is not None:
                active_slots[k] = self.slots[k]

        if "Frontal" not in active_slots:
            self.status_label_rt.configure(text="ERROR: FRONTAL SLOT REQUIRED FOR COMBINED CALCULATION", text_color=self.neon_magenta)
            self.deep_status_lbl.configure(text="Error: Frontal slot is empty or not rated.")
            return

        # Weights allocation
        weights = {"Frontal": 0.40}
        side_slots = [k for k in active_slots if k != "Frontal"]
        
        if side_slots:
            remaining_weight = 0.60
            w_each = remaining_weight / len(side_slots)
            for k in side_slots:
                weights[k] = w_each
        else:
            weights["Frontal"] = 1.0

        ai_score = sum(active_slots[k]["score"] * weights[k] for k in active_slots)
        geom_score = sum((active_slots[k]["geom_score"] or 0.0) * weights[k] for k in active_slots)
        symmetry = active_slots["Frontal"]["symmetry"] or 0.0
        golden_ratio = sum((active_slots[k]["golden_ratio"] or 0.0) * weights[k] for k in active_slots)

        # Potential calculation
        geom_factor = geom_score / 100.0
        potential_gain = (10.0 - ai_score) * (0.20 + 0.25 * geom_factor)
        potential_score = min(10.0, ai_score + potential_gain)

        # Update HUD Display with Consolidated Score
        self.score_label.configure(text=f"{ai_score:.2f} / 10.0", text_color=self.neon_green)
        tier_text, tier_color = self.get_tier_info(ai_score)
        self.tier_label.configure(text=tier_text, text_color=tier_color)
        top_pct = self.get_percentile_value(ai_score)
        self.percentile_label.configure(text=f"TOP: {top_pct:.1f}%", text_color=tier_color)
        self.potential_label.configure(text=f"POTENTIAL: {potential_score:.2f} / 10.0", text_color=self.neon_green)
        self.lbl_sym.configure(text=f"Combined Symmetry: {symmetry:.1f}%")
        self.pb_sym.set(symmetry / 100.0)
        self.lbl_gr.configure(text=f"Combined Golden Ratio: {golden_ratio:.1f}%")
        self.pb_gr.set(golden_ratio / 100.0)
        self.lbl_geom.configure(text=f"Combined Geometry: {geom_score:.1f}%")
        self.pb_geom.set(geom_score / 100.0)

        # Compile Consolidated Report
        report = []
        report.append("=== COMBINED CYBER-CERTIFICATE ===")
        report.append(f"Group: {self.group_combobox.get()}")
        report.append(f"Combined AI Rating: {ai_score:.2f}/10.0 (TOP: {top_pct:.1f}% of population)")
        report.append(f"Optimized Potential: {potential_score:.2f}/10.0")
        report.append(f"Combined Geometry: {geom_score:.1f}%")
        report.append("-" * 32)
        
        for k in active_slots:
            report.append(f"* {k}: {active_slots[k]['score']:.2f}/10 (Weight: {weights[k]*100:.0f}%)")
        report.append("-" * 32)
        
        if ai_score >= 8.0:
            verdict = "Outstanding proportions and aesthetics. High symmetry index."
        elif ai_score >= 6.5:
            verdict = "Excellent harmony of facial features. Good balance."
        elif ai_score >= 5.0:
            verdict = "Balanced structure. Minor deviations from ideals."
        else:
            verdict = "Pronounced asymmetry of contours. Specific facial relief."
            
        report.append(f"AI VERDICT: {verdict}")
        
        # Tailored improvement suggestions based on metrics
        report.append("-" * 32)
        report.append("=== IMPROVEMENT SUGGESTIONS ===")
        
        suggestions = []
        if symmetry < 90.0:
            suggestions.append("* Low symmetry: sleep on back, balance chewing on both sides.")
        
        for slot_name, slot_data in active_slots.items():
            for detail in slot_data.get("details", []):
                if "Match:" in detail:
                    try:
                        pct_str = detail.split("Match:")[1].split("%")[0].strip()
                        pct = float(pct_str)
                        if pct < 85.0:
                            if "Jaw" in detail and "* Reduce face bloat (body fat %) to define jawline." not in suggestions:
                                suggestions.append("* Reduce face bloat (body fat %) to define jawline.")
                            elif "Nose" in detail and "* Nose proportions off; groom eyebrows to balance spacing." not in suggestions:
                                suggestions.append("* Nose proportions off; groom eyebrows to balance spacing.")
                            elif "Height/Width" in detail and "* Head proportions: style hair with side volume/length." not in suggestions:
                                suggestions.append("* Head proportions: style hair with side volume/length.")
                    except Exception:
                        pass
                        
        if ai_score < 6.5:
            suggestions.append("* Skin & Styling: improve skin health, styling, and hair texture.")
            
        if not suggestions:
            suggestions.append("* Core geometry is optimized. Keep up current maintenance.")
            
        report.extend(suggestions)

        # Write logs
        self.log_box.configure(state="normal")
        self.log_box.delete("0.0", "end")
        self.log_box.insert("end", "\n".join(report))
        self.log_box.configure(state="disabled")

        self.log_to_console("\n" + "\n".join(report))
        
        self.status_label_rt.configure(text="COMBINED RATING CALCULATED", text_color=self.neon_green)
        self.deep_status_lbl.configure(text="Combined calculation completed!")

    def reset_slots(self):
        for key in self.slots:
            self.slots[key] = None
            
        # Reset buttons texts & styles
        self.btn_rt_frontal.configure(text="⬡ Frontal", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_rt_lsemi.configure(text="⬡ L.Semi", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_rt_rsemi.configure(text="⬡ R.Semi", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_rt_lprof.configure(text="⬡ L.Profile", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_rt_rprof.configure(text="⬡ R.Profile", fg_color="#101520", text_color=self.text_muted, border_width=0)

        self.btn_ph_frontal.configure(text="⬡ Slot 1: Frontal", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_ph_lsemi.configure(text="⬡ Slot 2: L. Semi-profile", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_ph_rsemi.configure(text="⬡ Slot 3: R. Semi-profile", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_ph_lprof.configure(text="⬡ Slot 4: L. Profile", fg_color="#101520", text_color=self.text_muted, border_width=0)
        self.btn_ph_rprof.configure(text="⬡ Slot 5: R. Profile", fg_color="#101520", text_color=self.text_muted, border_width=0)

        self.btn_combined_rt.configure(state="disabled")
        self.btn_run_deep.configure(state="disabled")
        self.status_label_rt.configure(text="STATUS: ALL SLOTS RESET", text_color=self.neon_cyan)
        
        self.photo_preview_label.configure(image=self.empty_ctk_img, text="NO PHOTO\n\nUpload photo for selected slot")
        self.tier_label.configure(text="", text_color=self.text_muted)
        self.percentile_label.configure(text="", text_color=self.text_muted)
        self.potential_label.configure(text="POTENTIAL: --.--", text_color=self.text_muted)
        self.deep_pb.set(0)
        self.deep_status_lbl.configure(text="Waiting for photo uploads...")

        # Reset console logs
        self.deep_log_box.configure(state="normal")
        self.deep_log_box.delete("0.0", "end")
        self.deep_log_box.insert("end", "AI console reset. Waiting for files...\n")
        self.deep_log_box.configure(state="disabled")

        if self.photo_webcam_active:
            self.toggle_photo_webcam()
        if self.current_frame is not None and self.webcam_running:
            self.process_and_display_frame(self.current_frame)
            
        self.select_photo_slot(self.selected_pose)

    def get_tier_info(self, score):
        if score is None or score <= 0:
            return "", self.text_muted
        if score < 3.0:
            return "SUB-3", "#ff3333"
        elif score < 4.0:
            return "SUB", "#ff6666"
        elif score < 5.0:
            return "LTN", "#ffcc00"
        elif score < 6.0:
            return "MTN", self.neon_cyan
        elif score < 7.0:
            return "HTN", self.neon_green
        elif score < 8.0:
            return "CHADLITE", "#bf00ff"
        else:
            return "CHAD", self.neon_magenta

    def get_percentile_value(self, score):
        import math
        mean = 5.0
        std = 1.15
        z = (score - mean) / std
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        top_pct = (1.0 - cdf) * 100.0
        return max(0.01, min(99.9, top_pct))

    def update_ui_with_metrics(self, metrics):
        if metrics["detected"]:
            if "ai_score" in metrics:
                self.score_label.configure(text=f"{metrics['ai_score']:.2f} / 10.0", text_color=self.neon_magenta)
                tier_text, tier_color = self.get_tier_info(metrics['ai_score'])
                self.tier_label.configure(text=tier_text, text_color=tier_color)
                top_pct = self.get_percentile_value(metrics['ai_score'])
                self.percentile_label.configure(text=f"TOP: {top_pct:.1f}%", text_color=tier_color)
                
                # Potential calculation
                overall_geom = metrics.get("overall_geom", 70.0)
                geom_factor = overall_geom / 100.0
                potential_gain = (10.0 - metrics['ai_score']) * (0.20 + 0.25 * geom_factor)
                potential_score = min(10.0, metrics['ai_score'] + potential_gain)
                self.potential_label.configure(text=f"POTENTIAL: {potential_score:.2f} / 10.0", text_color=self.neon_cyan)
            else:
                self.score_label.configure(text="LOADING...", text_color=self.neon_magenta)
                self.tier_label.configure(text="", text_color=self.text_muted)
                self.percentile_label.configure(text="", text_color=self.text_muted)
                self.potential_label.configure(text="POTENTIAL: LOADING...", text_color=self.text_muted)
            
            # Progress bars update
            self.lbl_sym.configure(text=f"Facial Symmetry: {metrics['symmetry']}%")
            self.pb_sym.set(metrics["symmetry"] / 100.0)
            
            self.lbl_gr.configure(text=f"Golden Ratio: {metrics['golden_ratio']}%")
            self.pb_gr.set(metrics["golden_ratio"] / 100.0)
            
            self.lbl_geom.configure(text=f"Geometry Index: {metrics['overall_geom']}%")
            self.pb_geom.set(metrics["overall_geom"] / 100.0)

            # Logs update
            self.log_box.configure(state="normal")
            self.log_box.delete("0.0", "end")
            
            if "ai_score" in metrics:
                self.log_box.insert("end", f"> [AI SCORE] Attractiveness: {metrics['ai_score']}/10.0\n")
                self.log_box.insert("end", f"> [POTENTIAL] Optimized: {potential_score:.2f}/10.0\n")
                self.log_box.insert("end", f"> [CLIP RAW] Aesthetic Index: {metrics['raw_score']}\n")
                self.log_box.insert("end", "-" * 38 + "\n")
                
            self.log_box.insert("end", "> [GEOMETRY] Analysis summary:\n")
            for detail in metrics["details"]:
                self.log_box.insert("end", f"* {detail}\n")
                
            # Quick suggestions in real-time scan
            if "ai_score" in metrics:
                self.log_box.insert("end", "-" * 38 + "\n")
                self.log_box.insert("end", "> [SUGGESTIONS] Looksmaxxing:\n")
                sugs = []
                if metrics["symmetry"] < 90.0:
                    sugs.append("* Balance chewing side & sleep posture")
                for d in metrics["details"]:
                    if "Match:" in d:
                        try:
                            val = float(d.split("Match:")[1].split("%")[0].strip())
                            if val < 85.0:
                                if "Jaw" in d:
                                    sugs.append("* Lower body fat % to define jawline")
                                elif "Nose" in d:
                                    sugs.append("* Groom eyebrows to balance facial ratios")
                        except Exception:
                            pass
                if metrics["ai_score"] < 6.5:
                    sugs.append("* Optimize skin hygiene & hair texture")
                
                if not sugs:
                    sugs.append("* Core geometry is optimized")
                for s in sugs[:2]: # Max 2 suggestions in real-time to save space
                    self.log_box.insert("end", f"{s}\n")
                    
            self.log_box.configure(state="disabled")
        else:
            self.score_label.configure(text="--.-- / 10.0", text_color=self.text_muted)
            self.tier_label.configure(text="", text_color=self.text_muted)
            self.percentile_label.configure(text="", text_color=self.text_muted)
            self.potential_label.configure(text="POTENTIAL: --.--", text_color=self.text_muted)
            self.lbl_sym.configure(text="Facial Symmetry: --%")
            self.pb_sym.set(0)
            self.lbl_gr.configure(text="Golden Ratio (1.618): --%")
            self.pb_gr.set(0)
            self.lbl_geom.configure(text="Overall Geometry Index: --%")
            self.pb_geom.set(0)
            
            self.log_box.configure(state="normal")
            self.log_box.delete("0.0", "end")
            self.log_box.insert("0.0", "No face detected in the scanning area...\n")
            self.log_box.configure(state="disabled")

    def display_image(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        ctk_img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(600, 400))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def toggle_telegram_bot(self):
        if self.bot_running:
            self.bot_running = False
            if self.bot_instance:
                self.bot_instance.stop()
                self.bot_instance = None
            self.btn_toggle_bot.configure(text="START BOT", fg_color="#1f2d3d", border_color=self.neon_green, hover_color=self.neon_green)
            self.lbl_bot_status.configure(text="BOT STATUS: INACTIVE", text_color=self.text_muted)
            self.log_to_console_on_gui("[GUI] Telegram bot stopped.")
        else:
            token = self.bot_token_entry.get().strip()
            if not token:
                self.lbl_bot_status.configure(text="BOT STATUS: ERROR (NO TOKEN)", text_color=self.neon_magenta)
                return
                
            self.lbl_bot_status.configure(text="BOT STATUS: STARTING...", text_color=self.neon_cyan)
            
            # Initialize CyberFaceBot
            from tg_bot import CyberFaceBot
            self.bot_instance = CyberFaceBot(
                token=token, 
                analyzer=self.analyzer, 
                predictor=self.predictor, 
                on_log_callback=self.log_to_console_on_gui
            )
            
            success = self.bot_instance.start()
            if success:
                self.bot_running = True
                self.btn_toggle_bot.configure(text="STOP BOT", fg_color="#1a1e29", border_color=self.neon_magenta, hover_color="#ff3333")
                self.lbl_bot_status.configure(text="BOT STATUS: ACTIVE", text_color=self.neon_green)
                self.log_to_console_on_gui("[GUI] Telegram bot started successfully.")
            else:
                self.lbl_bot_status.configure(text="BOT STATUS: ERROR", text_color=self.neon_magenta)
                self.bot_instance = None

    def log_to_console_on_gui(self, text):
        self.after(10, lambda: self._safe_log_to_gui(text))

    def _safe_log_to_gui(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def bind_entry_shortcuts(self, entry):
        # Bind standard English
        entry.bind("<Control-v>", lambda e: self.entry_paste(entry))
        entry.bind("<Control-V>", lambda e: self.entry_paste(entry))
        entry.bind("<Control-c>", lambda e: self.entry_copy(entry))
        entry.bind("<Control-C>", lambda e: self.entry_copy(entry))
        entry.bind("<Control-a>", lambda e: self.entry_select_all(entry))
        entry.bind("<Control-A>", lambda e: self.entry_select_all(entry))
        entry.bind("<Control-x>", lambda e: self.entry_cut(entry))
        entry.bind("<Control-X>", lambda e: self.entry_cut(entry))

        # Cyrillic bindings (wrapped in try-except to prevent bad keysym TclError crashes)
        cyrillic_bindings = [
            ("Cyrillic_m", self.entry_paste),
            ("Cyrillic_M", self.entry_paste),
            ("Cyrillic_es", self.entry_copy),
            ("Cyrillic_ES", self.entry_copy),
            ("Cyrillic_ef", self.entry_select_all),
            ("Cyrillic_EF", self.entry_select_all),
            ("Cyrillic_ch", self.entry_cut),
            ("Cyrillic_CH", self.entry_cut),
        ]
        for keysym, func in cyrillic_bindings:
            try:
                entry.bind(f"<Control-KeyPress-{keysym}>", lambda e, f=func: f(entry))
            except Exception:
                pass
            try:
                entry.bind(f"<Control-{keysym}>", lambda e, f=func: f(entry))
            except Exception:
                pass

    def entry_paste(self, entry):
        try:
            text = entry.clipboard_get()
            if entry.select_present():
                entry.delete("sel.first", "sel.last")
            entry.insert("insert", text)
        except Exception:
            pass
        return "break"

    def entry_copy(self, entry):
        try:
            if entry.select_present():
                text = entry.get()
                first = entry.index("sel.first")
                last = entry.index("sel.last")
                selected_text = text[first:last]
                entry.clipboard_clear()
                entry.clipboard_append(selected_text)
        except Exception:
            pass
        return "break"

    def entry_select_all(self, entry):
        entry.select_range(0, "end")
        entry.icursor("end")
        return "break"

    def entry_cut(self, entry):
        try:
            if entry.select_present():
                text = entry.get()
                first = entry.index("sel.first")
                last = entry.index("sel.last")
                selected_text = text[first:last]
                entry.clipboard_clear()
                entry.clipboard_append(selected_text)
                entry.delete("sel.first", "sel.last")
        except Exception:
            pass
        return "break"

    def destroy(self):
        if self.cap is not None:
            self.cap.release()
        if self.bot_running and self.bot_instance:
            self.bot_instance.stop()
        super().destroy()
