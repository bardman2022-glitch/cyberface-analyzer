import sys
from gui import CyberFaceApp

def main():
    try:
        app = CyberFaceApp()
        app.mainloop()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
