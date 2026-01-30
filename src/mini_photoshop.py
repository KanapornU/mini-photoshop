import cv2  # OpenCV: Used for image processing
import numpy as np  # NumPy: Used for pixel array manipulation
import tkinter as tk  # Tkinter: Main GUI library
from tkinter import filedialog, simpledialog, messagebox  # Tkinter modules for files, inputs, and alerts
from PIL import Image, ImageTk  # PIL: Converts OpenCV images for Tkinter display
import matplotlib.pyplot as plt  # Matplotlib: Used for drawing color histograms

class MiniPhotoshop:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop - Computer Vision Assignment")
        
        # Variables to store image data in OpenCV/NumPy array format
        self.cv_img = None
        self.original_img = None  # Backup for the Reset function

        # Initialize the Menu bar and UI
        self.setup_ui()
        
        # Create a Status Bar at the bottom
        self.info_var = tk.StringVar()
        self.info_var.set("Size: 0 x 0")
        self.status_bar = tk.Label(self.root, textvariable=self.info_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_ui(self):
        """ Function to create Menu Bar and Canvas for image display """
        menubar = tk.Menu(self.root)
        
        # 1. File Menu: For file management
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Reset Image", command=self.reset_image)
        menubar.add_cascade(label="File", menu=file_menu)

        # 2. Geometric Menu: Resize, Flip, Rotate, and Crop
        geo_menu = tk.Menu(menubar, tearoff=0)
        geo_menu.add_command(label="Enlarge/Shrink", command=self.resize_img)
        geo_menu.add_command(label="Flip Horizontal", command=lambda: self.flip_img(1))
        geo_menu.add_command(label="Flip Vertical", command=lambda: self.flip_img(0))
        geo_menu.add_command(label="Rotate 90 (CW/CCW)", command=self.rotate_90)
        geo_menu.add_command(label="Rotate Arbitrary", command=self.rotate_arbitrary)
        geo_menu.add_command(label="Crop (ROI Selection)", command=self.crop_image)
        menubar.add_cascade(label="Geometric", menu=geo_menu)

        # 3. Pixel & Color Menu: Color and pixel-level adjustments
        pixel_menu = tk.Menu(menubar, tearoff=0)
        pixel_menu.add_command(label="Grayscale", command=self.to_gray)
        pixel_menu.add_command(label="Invert Image", command=self.invert_img)
        pixel_menu.add_command(label="Brightness/Contrast", command=self.adjust_bc)
        pixel_menu.add_command(label="Histogram Equalization", command=self.hist_equal)
        pixel_menu.add_command(label="Adjust RGB Channels", command=self.adjust_rgb)
        pixel_menu.add_command(label="Adjust HSV Channels", command=self.adjust_hsv)
        menubar.add_cascade(label="Pixel/Color", menu=pixel_menu)

        # 4. Filtering Menu: Noise and image filters
        filter_menu = tk.Menu(menubar, tearoff=0)
        filter_menu.add_command(label="Add Salt-and-Pepper Noise", command=self.add_sp_noise)
        filter_menu.add_command(label="Add Gaussian Noise", command=self.add_gaussian_noise)
        filter_menu.add_command(label="Blur (Box Filter)", command=lambda: self.apply_blur('box'))
        filter_menu.add_command(label="Blur (Gaussian Filter)", command=lambda: self.apply_blur('gauss'))
        filter_menu.add_command(label="Blur (Median Filter)", command=lambda: self.apply_blur('median'))
        filter_menu.add_command(label="Sharpen Image", command=self.sharpen_img)
        filter_menu.add_command(label="Edge Detection (Canny)", command=self.edge_detect)
        menubar.add_cascade(label="Filtering", menu=filter_menu)

        # 5. Arithmetic & Info Menu: Image math and statistics
        arith_menu = tk.Menu(menubar, tearoff=0)
        arith_menu.add_command(label="Add/Subtract Image", command=self.arithmetic_op)
        arith_menu.add_command(label="Blend Images (Transparency)", command=self.blend_images)
        arith_menu.add_command(label="Show Size & RGB Histogram", command=self.show_info_hist)
        menubar.add_cascade(label="Arithmetic/Info", menu=arith_menu)

        self.root.config(menu=menubar)
        
        # Image Display Canvas
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="#333333")
        self.canvas.pack(pady=10)

    # --- FILE FUNCTIONS ---
    def load_image(self):
        """ Open a file dialog to load an image """
        path = filedialog.askopenfilename()
        if path:
            self.cv_img = cv2.imread(path)  # Read image with OpenCV
            self.original_img = self.cv_img.copy()  # Backup for Reset
            self.update_canvas()

    def save_image(self):
        """ Save the edited image to disk """
        if self.cv_img is not None:
            path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if path: cv2.imwrite(path, self.cv_img)

    def reset_image(self):
        """ Revert the image to its original state """
        if self.original_img is not None:
            self.cv_img = self.original_img.copy()
            self.update_canvas()

    def update_canvas(self):
        """ Update the image display on the GUI canvas """
        if self.cv_img is not None:
            # Update image dimensions in Status Bar
            h, w = self.cv_img.shape[:2]
            self.info_var.set(f" Image Size: {w} (Width) x {h} (Height) px")

            display_img = self.cv_img.copy()
            
            # Convert color space: OpenCV uses BGR, but PIL/Tkinter uses RGB
            if len(display_img.shape) == 2:  # For Grayscale images
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            else:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy Array to ImageTk format
            img = Image.fromarray(display_img)
            img.thumbnail((780, 580))  # Resize preview to fit canvas while keeping aspect ratio
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(400, 300, image=self.tk_img)

    # --- GEOMETRIC FUNCTIONS ---
    def resize_img(self):
        """ Resize the image based on a scale factor """
        s = simpledialog.askfloat("Resize", "Enter scale factor (e.g. 0.5 to shrink, 1.5 to enlarge):")
        if s: 
            self.cv_img = cv2.resize(self.cv_img, None, fx=s, fy=s)
            self.update_canvas()

    def flip_img(self, mode):
        """ Flip image (1 = Horizontal, 0 = Vertical) """
        self.cv_img = cv2.flip(self.cv_img, mode)
        self.update_canvas()

    def rotate_90(self):
        """ Rotate image 90 degrees clockwise """
        self.cv_img = cv2.rotate(self.cv_img, cv2.ROTATE_90_CLOCKWISE)
        self.update_canvas()

    def rotate_arbitrary(self):
        """ Rotate image by a user-defined angle (Rotation Matrix) """
        angle = simpledialog.askfloat("Rotate", "Enter angle (degrees):")
        if angle:
            h, w = self.cv_img.shape[:2]
            # Get center point and rotation matrix
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            # Apply warpAffine to rotate the image
            self.cv_img = cv2.warpAffine(self.cv_img, M, (w, h))
            self.update_canvas()

    def crop_image(self):
        """ Select a Region of Interest (ROI) and crop the image """
        if self.cv_img is None: return
        
        messagebox.showinfo("Instruction", "Select the area with your mouse.\nPress ENTER or SPACE to confirm.")
        window_name = "Select Region to Crop"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # OpenCV function to select area using mouse
        roi = cv2.selectROI(window_name, self.cv_img, fromCenter=False, showCrosshair=True)
        
        if roi != (0,0,0,0):
            x, y, w, h = roi
            # Crop image using NumPy slicing
            self.cv_img = self.cv_img[int(y):int(y+h), int(x):int(x+w)]
            self.update_canvas()
            
        cv2.destroyWindow(window_name)

    # --- PIXEL & COLOR FUNCTIONS ---
    def to_gray(self):
        """ Convert BGR color image to Grayscale """
        if len(self.cv_img.shape) == 3:
            self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
            self.update_canvas()

    def invert_img(self):
        """ Invert image colors (Negative) using Bitwise Not """
        self.cv_img = cv2.bitwise_not(self.cv_img)
        self.update_canvas()

    def adjust_bc(self):
        """ Basic Brightness (Beta) and Contrast (Alpha) adjustment """
        self.cv_img = cv2.convertScaleAbs(self.cv_img, alpha=1.2, beta=30)
        self.update_canvas()

    def hist_equal(self):
        """ Improve image exposure using Histogram Equalization """
        if len(self.cv_img.shape) == 2:
            self.cv_img = cv2.equalizeHist(self.cv_img)
        else:
            # For color images, convert to YUV to adjust brightness (Y channel) only
            img_yuv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            self.cv_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.update_canvas()

    def adjust_rgb(self):
        """ Example: Split channels and boost Blue intensity """
        b, g, r = cv2.split(self.cv_img)
        b = cv2.add(b, 50) 
        self.cv_img = cv2.merge((b, g, r))
        self.update_canvas()

    def adjust_hsv(self):
        """ Increase color Saturation in HSV color space """
        hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 50) 
        self.cv_img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        self.update_canvas()

    # --- FILTERING FUNCTIONS ---
    def add_sp_noise(self):
        """ Add Salt-and-Pepper noise (black and white dots) """
        prob = 0.02
        noise = np.zeros(self.cv_img.shape, np.uint8)
        cv2.randu(noise, 0, 255)
        self.cv_img[noise < (prob * 255)] = 0
        self.cv_img[noise > (255 - prob * 255)] = 255
        self.update_canvas()

    def add_gaussian_noise(self):
        """ Add random noise following a Gaussian distribution """
        gauss = np.random.normal(0, 15, self.cv_img.shape).astype(np.uint8)
        self.cv_img = cv2.add(self.cv_img, gauss)
        self.update_canvas()

    def apply_blur(self, btype):
        """ Apply different types of Blur filters """
        if btype == 'box': self.cv_img = cv2.blur(self.cv_img, (5,5))
        elif btype == 'gauss': self.cv_img = cv2.GaussianBlur(self.cv_img, (5,5), 0)
        elif btype == 'median': self.cv_img = cv2.medianBlur(self.cv_img, 5)
        self.update_canvas()

    def sharpen_img(self):
        """ Enhance image details using a Sharpening Kernel (2D Filter) """
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.cv_img = cv2.filter2D(self.cv_img, -1, kernel)
        self.update_canvas()

    def edge_detect(self):
        """ Extract image edges using Canny Edge Detection """
        self.cv_img = cv2.Canny(self.cv_img, 100, 200)
        self.update_canvas()

    # --- ARITHMETIC & INFO FUNCTIONS ---
    def arithmetic_op(self):
        """ Add two images together (increases brightness) """
        path = filedialog.askopenfilename(title="Select second image for ADD")
        if path:
            img2 = cv2.imread(path)
            img2 = cv2.resize(img2, (self.cv_img.shape[1], self.cv_img.shape[0])) # Must be the same size
            self.cv_img = cv2.add(self.cv_img, img2)
            self.update_canvas()

    def blend_images(self):
        """ Combine two images with transparency (Alpha Blending) """
        path = filedialog.askopenfilename(title="Select image to blend")
        if path:
            img2 = cv2.imread(path)
            img2 = cv2.resize(img2, (self.cv_img.shape[1], self.cv_img.shape[0]))
            # 0.6 and 0.4 represent the weight (opacity) of each image
            self.cv_img = cv2.addWeighted(self.cv_img, 0.6, img2, 0.4, 0)
            self.update_canvas()

    def show_info_hist(self):
        """ Display RGB Histograms to analyze color distribution """
        color = ('b','g','r')
        plt.figure("3-Channel Histogram")
        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity (0-255)")
        plt.ylabel("Number of Pixels") 
        
        for i, col in enumerate(color):
            # Calculate Histogram for each color channel
            hist = cv2.calcHist([self.cv_img], [i], None, [256], [0,256])
            plt.plot(hist, color=col)
            plt.xlim([0,256])
        plt.show()  # Display graph window

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()