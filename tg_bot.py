import os
import threading
import time
import math
import subprocess
import cv2
import numpy as np
from PIL import Image
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

class CyberFaceBot:
    def __init__(self, token, analyzer, predictor, on_log_callback=None):
        self.token = token
        self.analyzer = analyzer
        self.predictor = predictor
        self.on_log_callback = on_log_callback
        
        self.bot = None
        self.thread = None
        self.is_running = False
        
        # FastAPI backend uvicorn thread
        self.server_thread = None
        
        # SSH Tunnel subprocess
        self.tunnel_process = None
        self.tunnel_thread = None
        self.tunnel_url = None
        
        # User configurations (chat_id -> target_group)
        self.user_groups = {}

    def log(self, message):
        if self.on_log_callback:
            self.on_log_callback(f"[Bot] {message}")
        else:
            print(f"[Bot] {message}")

    def start(self):
        if self.is_running:
            return False
            
        try:
            # 1. Start FastAPI Backend
            self.start_backend()
            
            # 2. Establish SSH Tunnel
            self.start_tunnel()
            
            # 3. Start Telebot Instance
            self.bot = telebot.TeleBot(self.token, threaded=False)
            self.register_handlers()
            self.is_running = True
            
            # Start Telegram Bot Polling in a background thread
            self.thread = threading.Thread(target=self.poll_loop, daemon=True)
            self.thread.start()
            self.log("Bot started successfully and polling.")
            return True
        except Exception as e:
            self.log(f"Error starting bot: {e}")
            self.stop()
            return False

    def start_backend(self):
        # Inject models into tma_backend
        import tma_backend
        tma_backend.analyzer = self.analyzer
        tma_backend.predictor = self.predictor
        
        import uvicorn
        self.server_thread = threading.Thread(
            target=lambda: uvicorn.run(tma_backend.app, host="127.0.0.1", port=23789, log_level="warning"),
            daemon=True
        )
        self.server_thread.start()
        self.log("FastAPI backend server started on localhost:23789.")

    def start_tunnel(self):
        # Ensure SSH key exists to allow localhost.run tunnel to connect
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        if not os.path.exists(ssh_key_path):
            self.log("SSH key not found. Generating a secure keypair automatically...")
            ssh_dir = os.path.dirname(ssh_key_path)
            if not os.path.exists(ssh_dir):
                os.makedirs(ssh_dir)
            try:
                subprocess.run(
                    ["ssh-keygen", "-t", "rsa", "-b", "2048", "-N", "", "-f", ssh_key_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                self.log("SSH keypair generated successfully.")
            except Exception as e:
                self.log(f"Failed to generate SSH key automatically: {e}")

        self.log("Starting SSH tunnel to localhost.run...")
        try:
            self.tunnel_process = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:127.0.0.1:23789", "nokey@localhost.run"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            def read_stdout():
                for line in iter(self.tunnel_process.stdout.readline, ""):
                    if not self.is_running:
                        break
                    # Look for URL
                    if "https://" in line:
                        parts = line.split("https://")
                        if len(parts) > 1:
                            url = "https://" + parts[1].strip().split(" ")[0].split("\n")[0]
                            self.tunnel_url = url
                            self.log(f"Public HTTPS Tunnel established: {self.tunnel_url}")
                    # Pipe line to console if needed
                    # self.log(f"Tunnel Log: {line.strip()}")
                    
            self.tunnel_thread = threading.Thread(target=read_stdout, daemon=True)
            self.tunnel_thread.start()
        except Exception as e:
            self.log(f"Failed to start SSH tunnel: {e}")

    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        
        # 1. Stop Telegram bot polling
        if self.bot:
            try:
                self.bot.stop_polling()
            except Exception as e:
                self.log(f"Error stopping polling: {e}")
                
        # 2. Terminate SSH tunnel process
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=2)
                self.log("SSH Tunnel process terminated.")
            except Exception as e:
                self.log(f"Error terminating SSH Tunnel: {e}")
            self.tunnel_process = None
            
        self.tunnel_url = None
        self.log("Bot and services stopped.")

    def poll_loop(self):
        while self.is_running:
            try:
                self.bot.polling(non_stop=True, timeout=10, long_polling_timeout=5)
            except Exception as e:
                self.log(f"Polling exception: {e}")
                time.sleep(2)

    def register_handlers(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            chat_id = message.chat.id
            self.user_groups[chat_id] = "Universal"
            
            # Wait briefly if tunnel is still establishing
            retries = 3
            while not self.tunnel_url and retries > 0:
                time.sleep(1.0)
                retries -= 1
                
            welcome_text = (
                "🤖 *Welcome to CyberFace Analyzer Bot!*\n\n"
                "Send me a selfie or portrait photo, and I will analyze your face geometry, "
                "symmetry, and calculate an AI beauty rating based on our deep neural networks.\n\n"
                "⚡ *Telegram Mini App is now active!* Tap the button below to open the interactive interface, "
                "upload multiple photos, and compile a weighted looksmaxxing certificate!\n\n"
                "Current calibration: *Universal*\n\n"
                "Use the buttons below to change target calibration group:"
            )
            
            markup = self.create_group_markup()
            self.bot.send_message(chat_id, welcome_text, parse_mode="Markdown", reply_markup=markup)

        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("setgroup_"))
        def handle_group_change(call):
            chat_id = call.message.chat.id
            group = call.data.split("setgroup_")[1]
            self.user_groups[chat_id] = group
            
            self.bot.answer_callback_query(call.id, f"Group set to {group}")
            self.bot.send_message(chat_id, f"🎯 Calibration target changed to *{group}*.\nSend a photo to begin rating!", parse_mode="Markdown")

        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            chat_id = message.chat.id
            group = self.user_groups.get(chat_id, "Universal")
            
            self.bot.send_chat_action(chat_id, 'upload_photo')
            self.log(f"Received photo from chat_id {chat_id}. Processing under calibration: {group}")
            
            try:
                # 1. Download photo
                file_info = self.bot.get_file(message.photo[-1].file_id)
                downloaded_file = self.bot.download_file(file_info.file_path)
                
                # 2. Decode into cv2 BGR image
                nparr = np.frombuffer(downloaded_file, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    self.bot.reply_to(message, "❌ Failed to decode image file.")
                    return
                
                # 3. Resize and pad image to 640x480 for standardized analysis
                h, w = img.shape[:2]
                scale = min(640 / w, 480 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                padded_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                dx = (640 - new_w) // 2
                dy = (480 - new_h) // 2
                padded_frame[dy:dy+new_h, dx:dx+new_w] = img_resized
                
                # 4. Analyze frame
                hud_frame, face_crop, metrics = self.analyzer.analyze_frame(padded_frame, target_group=group, draw_hud=True)
                
                if not metrics["detected"] or face_crop is None:
                    self.bot.reply_to(message, "❌ No face detected. Please send a clear, forward-facing photo with good lighting.")
                    return
                
                # 5. Predict beauty score
                geom_data = {
                    "symmetry": metrics.get("symmetry", 0.0),
                    "golden_ratio": metrics.get("golden_ratio", 0.0),
                    "overall_geom": metrics.get("overall_geom", 0.0)
                }
                
                score_10, raw_score = self.predictor.predict(face_crop, target_group=group, is_webcam=False, geom_data=geom_data)
                if score_10 is None:
                    score_10 = 0.0
                    raw_score = 0.0
                
                metrics["ai_score"] = score_10
                metrics["raw_score"] = raw_score
                
                # 6. Format certificate text
                report = self.compile_report(group, metrics)
                
                # 7. Send back annotated image and report text
                _, buffer = cv2.imencode('.jpg', hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_bytes = buffer.tobytes()
                
                self.bot.send_photo(
                    chat_id, 
                    img_bytes, 
                    caption=report, 
                    parse_mode="Markdown", 
                    reply_markup=self.create_group_markup()
                )
                
            except Exception as e:
                self.log(f"Exception handling photo: {e}")
                self.bot.reply_to(message, "❌ An error occurred during image processing.")

    def create_group_markup(self):
        markup = InlineKeyboardMarkup()
        if self.tunnel_url:
            markup.row(InlineKeyboardButton("💻 OPEN MINI APP", web_app=telebot.types.WebAppInfo(url=self.tunnel_url)))
            
        btn1 = InlineKeyboardButton("Universal", callback_data="setgroup_Universal")
        btn2 = InlineKeyboardButton("Strict Accuracy", callback_data="setgroup_Strict")
        btn3 = InlineKeyboardButton("Young Man (14-20)", callback_data="setgroup_Young Man")
        btn4 = InlineKeyboardButton("Man (21+)", callback_data="setgroup_Man")
        btn5 = InlineKeyboardButton("Young Woman (14-20)", callback_data="setgroup_Young Woman")
        btn6 = InlineKeyboardButton("Woman (21+)", callback_data="setgroup_Woman")
        
        markup.row(btn1, btn2)
        markup.row(btn3, btn4)
        markup.row(btn5, btn6)
        return markup

    def compile_report(self, group, metrics):
        score = metrics["ai_score"]
        raw = metrics["raw_score"]
        sym = metrics["symmetry"]
        gr = metrics["golden_ratio"]
        geom = metrics["overall_geom"]
        
        # Looksmaxxing Tier
        is_female = "Woman" in group
        def get_tier_info(s):
            if s < 3.0: return "SUB-3", "❌ Low Harmony"
            elif s < 4.0: return "SUB", "❌ Below Average"
            elif s < 5.0: return "LTB" if is_female else "LTN", "⚠️ Lite-Average"
            elif s < 6.0: return "MTB" if is_female else "MTN", "🌐 Mid-Average"
            elif s < 7.0: return "HTB" if is_female else "HTN", "✅ High-Average"
            elif s < 8.0: return "STACYLITE" if is_female else "CHADLITE", "⚡ Stacylite (Excellent)" if is_female else "⚡ Chadlite (Excellent)"
            else: return "STACY" if is_female else "CHAD", "🔥 Stacy (Elite)" if is_female else "🔥 Chad/Stacy (Elite)"
            
        tier_text, tier_desc = get_tier_info(score)
        
        # Percentile
        mean = 5.0
        std = 1.15
        z = (score - mean) / std
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        top_pct = (1.0 - cdf) * 100.0
        top_pct = max(0.01, min(99.9, top_pct))
        
        # Potential calculation
        geom_factor = geom / 100.0
        potential_gain = (10.0 - score) * (0.20 + 0.25 * geom_factor)
        potential_score = min(10.0, score + potential_gain)
        
        report = (
            f"=== 🤖 *CYBER-CERTIFICATE REPORT* ===\n"
            f"• *Calibration:* {group}\n"
            f"• *AI Attractiveness Rating:* `{score:.2f}/10.0`\n"
            f"• *Looksmaxxing Tier:* `{tier_text}` ({tier_desc})\n"
            f"• *Percentile:* `TOP {top_pct:.1f}%` of population\n"
            f"• *Optimized Potential:* `{potential_score:.2f}/10.0`\n"
            f"--------------------------------------\n"
            f"📐 *GEOMETRIC SYMMETRY ANALYSIS*\n"
            f"• *Facial Symmetry:* `{sym:.1f}%`\n"
            f"• *Golden Ratio:* `{gr:.1f}%`\n"
            f"• *Overall Geometry:* `{geom:.1f}%`\n"
            f"--------------------------------------\n"
            f"💡 *LOOKSMAXXING SUGGESTIONS:*\n"
        )
        
        sugs = []
        if sym < 90.0:
            sugs.append("* Balance chewing side & sleep posture.")
        metrics_details = metrics.get("details", [])
        for d in metrics_details:
            if "Match:" in d:
                try:
                    pct_str = d.split("Match:")[1].split("%")[0].strip()
                    pct = float(pct_str)
                    if pct < 85.0:
                        if "Jaw" in d and "* Lower body fat % to define jawline." not in sugs:
                            sugs.append("* Lower body fat % to define jawline.")
                        elif "Nose" in d and "* Groom eyebrows to balance ratio." not in sugs:
                            sugs.append("* Groom eyebrows to balance ratio.")
                except:
                    pass
        if score < 6.5:
            sugs.append("* Optimize skin hygiene & hair texture.")
        if not sugs:
            sugs.append("* Core geometry is optimized.")
            
        for s in sugs[:3]:
            report += f"{s}\n"
            
        report += "======================================"
        return report
