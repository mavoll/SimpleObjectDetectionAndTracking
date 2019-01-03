import cv2
import imutils
from imutils.object_detection import non_max_suppression
from imutils.video import FPS
import numpy as np
import sys
from threading import Thread
import time
from tkinter import BooleanVar
from tkinter import Button
from tkinter import Checkbutton
from tkinter import Entry
from tkinter.filedialog import askopenfilename
from tkinter import IntVar
from tkinter import Label
from tkinter import messagebox
from tkinter import Tk
from tkinter import Toplevel
from tkinter import ttk
from tkinter import W


class App(object):

    face_fn = "haarcascade_xml/haarcascade_frontalface_default.xml"
    left_eye_fn = "haarcascade_xml/haarcascade_eye.xml"
    mouth_fn = "haarcascade_xml/haarcascade_mcs_mouth.xml"
    upperbody_fn = "haarcascade_xml/haarcascade_upperbody.xml"
    lowerbody_fn = "haarcascade_xml/haarcascade_lowerbody.xml"
    fullbody_fn = "haarcascade_xml/haarcascade_fullbody.xml"
    eye_tree_eyeglasses_fn = "haarcascade_xml/haarcascade_eye_tree_eyeglasses.xml"
    mcs_upperbody_fn = "haarcascade_xml/haarcascade_mcs_upperbody.xml"
    profileface_fn = "haarcascade_xml/haarcascade_profileface.xml"
    car_fn = "haarcascade_xml/haarcascade_car.xml"
    head_fn = "haarcascade_xml/haarcascade_head.xml"
    people_fn = "haarcascade_xml/haarcascade_people.xml"

    def __init__(self):

        self.root = Tk()
        self.scale_factor = 0.7
        self.cam = None
        self.cam_id = 0
        self.input_source = None
        self.source_changed = False
        self.opencv_thread = None

        self.img = None
        self.selection = None
        self.drag_start = None

        self.detection_methods = {'HOG SVN default people': BooleanVar(value=True), 'Haar Face': BooleanVar(value=False), 'Haar left eye': BooleanVar(value=False),
                                  'Haar mouth': BooleanVar(value=False), 'Haar upper body': BooleanVar(value=False), 'Haar lower body': BooleanVar(value=False),
                                  'Haar full body': BooleanVar(value=False), 'Haar eye tree glasses': BooleanVar(value=False), 'Haar mcs upper body': BooleanVar(value=False),
                                  'Haar face profile': BooleanVar(value=False), 'Haar car': BooleanVar(value=False), 'Haar head': BooleanVar(value=False), 'Haar people': BooleanVar(value=True)}

        self.tracking_methods = {'Camshift': BooleanVar(value=False), 'MIL': BooleanVar(value=False), 'KCF': BooleanVar(value=True), 'CSRT': BooleanVar(value=False),
                                 'Boosting': BooleanVar(value=False), 'TLD': BooleanVar(value=False), 'Median flow': BooleanVar(value=False), 'MOOSE': BooleanVar(value=False)}

        # HOG parameter
        self.winStride = (8, 8)
        self.roi_padding = (16, 16)
        self.pyramid_scale = 1.05
        self.meanShift = 0
        # Cascade Detection parameter
        self.haar_scale_factor = 1.05
        self.haar_min_neighbors = 2
        self.haar_min_size = (40, 40)
        self.haar_max_size = (80, 80)

        # Camshift termination parameter
        self.max_iter = 10
        self.min_movement = 1

        self.show_mask_image = False

        self.hue_max = 180
        self.hue_min = 0
        self.sat_max = 255
        self.sat_min = 60
        self.val_max = 255
        self.val_min = 20

        self.show_bgr_mask_image = False
        self.blue_max = 255
        self.blue_min = 50
        self.green_max = 130
        self.green_min = 118
        self.red_max = 130
        self.red_min = 30

        self.show_lab_mask_image = False
        self.lig_max = 72
        self.lig_min = 0
        self.gm_max = 255
        self.gm_min = 0
        self.by_max = 255
        self.by_min = 0

        self.show_hsv_mask_conjunction = True

        self.mask_to_use = 'HSV'
        self.mask = None

        self.show_hsv_hist = True
        self.show_bgr_hist = False
        self.show_lab_hist = False

        self.use_non_max_suppression = True
        self.overlapThresh = 0.4

    def run(self):

        self.root.wm_title("Select actions:")
        self.root.resizable(width=False, height=False)
        self.root.geometry('{}x{}'.format(250, 650))
        self.root.attributes("-topmost", True)

        btn1 = Button(self.root, text="Open video file", command=self.open_video)
        btn1.pack(side="top", fill="both", expand="yes", padx="10", pady="5")

        labelTop3 = Label(self.root,
                          text="Choose cam id")
        labelTop3.pack(side="top", padx="10", pady="5")

        comboExample3 = ttk.Combobox(self.root,
                                     values=[
                                         0,
                                         1,
                                         2])

        comboExample3.current(0)
        comboExample3.state(['readonly'])
        comboExample3.bind("<<ComboboxSelected>>", self.set_cam_id)
        comboExample3.pack(side="top", padx="10", pady="5")

        btn2 = Button(self.root, text="Use Webcam", command=self.open_webcam)
        btn2.pack(side="top", fill="both", expand="yes", padx="10", pady="5")

        labelTop = Label(self.root,
                         text="Choose video scale factor")
        labelTop.pack(side="top", padx="10", pady="5")

        comboExample = ttk.Combobox(self.root,
                                    values=[
                                        0.5,
                                        0.6,
                                        0.7,
                                        0.8,
                                        0.9,
                                        1.0])

        comboExample.current(2)
        comboExample.state(['readonly'])
        comboExample.bind("<<ComboboxSelected>>", self.set_scale_factor)
        comboExample.pack(side="top", padx="10", pady="5")

        labelTop6 = Label(self.root,
                          text="Masks")
        labelTop6.pack(side="top", padx="10", pady="5")

        self.v_mask = IntVar()
        self.v_mask.set(self.show_mask_image)
        Checkbutton(self.root,
                    text="Show HSV mask?",
                    padx=20,
                    variable=self.v_mask,
                    command=self.set_show_mask).pack(side="top", anchor=W, padx="5", pady="5")

        self.v_BGR_mask = IntVar()
        self.v_BGR_mask.set(self.show_bgr_mask_image)
        Checkbutton(self.root,
                    text="Show BGR mask?",
                    padx=20,
                    variable=self.v_BGR_mask,
                    command=self.set_bgr_show_mask).pack(side="top", anchor=W, padx="5", pady="5")

        self.v_Lab_mask = IntVar()
        self.v_Lab_mask.set(self.show_lab_mask_image)
        Checkbutton(self.root,
                    text="Show Lab mask?",
                    padx=20,
                    variable=self.v_Lab_mask,
                    command=self.set_lab_show_mask).pack(side="top", anchor=W, padx="5", pady="5")

        self.v_mask_conj = IntVar()
        self.v_mask_conj.set(self.show_hsv_mask_conjunction)
        Checkbutton(self.root,
                    text="Show mask as conjunction?",
                    padx=20,
                    variable=self.v_mask_conj,
                    command=self.set_show_mask_conj).pack(side="top", anchor=W, padx="5", pady="5")

        labelTop4 = Label(self.root,
                          text="Which mask to use for det.?")
        labelTop4.pack(side="top", padx="10", pady="5")

        comboExample4 = ttk.Combobox(self.root,
                                     values=[
                                         'BGR mask',
                                         'HSV mask',
                                         'Lab mask'])

        comboExample4.current(1)
        comboExample4.state(['readonly'])
        comboExample4.bind("<<ComboboxSelected>>", self.set_mask)
        comboExample4.pack(side="top", padx="10", pady="5")

        labelTop5 = Label(self.root,
                          text="Histograms in regards to selected area?")
        labelTop5.pack(side="top", padx="10", pady="5")

        self.v_hsvhist = IntVar()
        self.v_hsvhist.set(self.show_hsv_hist)
        Checkbutton(self.root,
                    text="Show hsv histogram?",
                    padx=20,
                    variable=self.v_hsvhist,
                    command=self.set_show_hsv_hist).pack(side="top", anchor=W, padx="5", pady="5")

        self.v_bgrhist = IntVar()
        self.v_bgrhist.set(self.show_bgr_hist)
        Checkbutton(self.root,
                    text="Show bgr histogram?",
                    padx=20,
                    variable=self.v_bgrhist,
                    command=self.set_show_bgr_hist).pack(side="top", anchor=W, padx="5", pady="5")

        self.v_labhist = IntVar()
        self.v_labhist.set(self.show_lab_hist)
        Checkbutton(self.root,
                    text="Show lab histogram?",
                    padx=20,
                    variable=self.v_labhist,
                    command=self.set_show_lab_hist).pack(side="top", anchor=W, padx="5", pady="5")

        btn_detect = Button(self.root, text="Detection options",
                            command=self.ask_for_detection_options)
        btn_detect.pack(side="top", fill="both", expand="yes", padx="10", pady="5")

        btn_track = Button(self.root, text="Tracking options",
                           command=self.ask_for_tracking_options)
        btn_track.pack(side="top", fill="both", expand="yes", padx="10", pady="5")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

        cv2.destroyAllWindows()
        sys.exit()

    def open_webcam(self):

        if self.cam_id is not None:
            self.input_source = self.cam_id
            self.source_changed = True
            self.start_video()

    def open_video(self):

        options = {}
        options['title'] = "Choose video"

        filename = askopenfilename(**options)

        if filename:
            self.input_source = filename
            self.source_changed = True
            self.start_video()

    def start_video(self):

        if self.opencv_thread is None:
            self.source_changed = False
            self.opencv_thread = Thread(target=self.run_opencv_thread)
            self.opencv_thread.daemon = True
            self.opencv_thread.start()

    def run_opencv_thread(self):

        cv2.namedWindow('source')
        cv2.setMouseCallback('source', self.onmouse)

        self.start_processing()

        cv2.destroyAllWindows()
        self.opencv_thread = None

    def start_processing(self):

        if self.input_source is not None:

            self.cam = cv2.VideoCapture(self.input_source)

            if self.cam.isOpened():

                #                self.cam.set(cv2.CAP_PROP_FPS,1)
                rects = []
                fps = None
                fps = FPS().start()
                multi_tracker = None

                while not self.source_changed and self.cam.isOpened():

                    try:

                        ret, self.img = self.cam.read()

                        if ret is True:

                            h = int(self.img.shape[0] * self.scale_factor)
                            w = int(self.img.shape[1] * self.scale_factor)
                            self.img = cv2.resize(self.img, (w, h))

                            vis = self.img.copy()
    #                        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                            hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
                            lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                            mask = self.show_masks(vis, hsv, lab)

                            if self.selection and (self.selection[2] - self.selection[0]) > 0 and (self.selection[3] - self.selection[1]) > 0:

                                s_x0, s_y0, s_x1, s_y1 = self.selection
                                hsv_roi = hsv[s_y0:s_y1, s_x0:s_x1]
                                vis_roi = vis[s_y0:s_y1, s_x0:s_x1]
                                lab_roi = lab[s_y0:s_y1, s_x0:s_x1]

                                if self.drag_start:

                                    rects = []

                                    self.show_hists(vis_roi, hsv_roi, lab_roi)

                                    cv2.rectangle(vis, (s_x0, s_y0), (s_x1, s_y1), (255, 0, 0), 1)
                                    cv2.imshow('source', vis)

                                    if self.detection_methods.get('Haar Face').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.face_rects_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar mcs upper body').get() is True:
                                        rec = self.detect_haar_cascade(
                                            vis_roi, App.mcs_upperbody_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar left eye').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.left_eye_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar people').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.people_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar head').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.head_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar car').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.car_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar upper body').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.upperbody_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar full body').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.fullbody_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar lower body').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.lowerbody_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('Haar face profile').get() is True:
                                        rec = self.detect_haar_cascade(vis_roi, App.profileface_fn)
                                        if rec is not None:
                                            rects.extend(rec)
                                    if self.detection_methods.get('HOG SVN default people').get() is True:
                                        rec = self.detect_hog_svn(vis_roi)
                                        if rec is not None:
                                            rects.extend(rec)

                                    if len(rects) > 0:

                                        s_x0, s_y0, s_x1, s_y1 = self.selection

                                        if self.use_non_max_suppression:
                                            rects = np.array([[x, y, x + w, y + h]
                                                              for (x, y, w, h) in rects])
                                            rects = non_max_suppression(
                                                rects, probs=None, overlapThresh=self.overlapThresh)
                                            rects = np.array([[x, y, w - x, h - y]
                                                              for (x, y, w, h) in rects])

                                        multi_tracker = cv2.MultiTracker_create()

                                        for i, (x, y, w, h) in enumerate(rects):

                                            rects[i] = [x + s_x0, y + s_y0, w, h]

                                            cv2.rectangle(vis, ((x + s_x0), (y + s_y0)),
                                                          (((x + w) + s_x0), ((y + h) + s_y0)), (255, 0, 0), 2)

                                            if self.tracking_methods.get('MIL').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerMIL_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('KCF').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerKCF_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('CSRT').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerCSRT_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('Boosting').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerBoosting_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('TLD').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerTLD_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('Median flow').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerMedianFlow_create(), vis, (x + s_x0, y + s_y0, w, h))

                                            if self.tracking_methods.get('MOOSE').get() is True:
                                                multi_tracker.add(
                                                    cv2.TrackerMOSSE_create(), vis, (x + s_x0, y + s_y0, w, h))

                                        cv2.imshow('source', vis)

                                else:

                                    if rects != []:

                                        if self.tracking_methods.get('Camshift').get() is True:

                                            for i, (x00, y00, w, h) in enumerate(rects):

                                                hsv_det = hsv[y00:(y00 + h), x00:(x00 + w)]
                                                mask_det = mask[y00:(y00 + h), x00:(x00 + w)]
                                                hist = cv2.calcHist(
                                                    [hsv_det], [0], mask_det, [16], [0, 180])
                                                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

                                                rec, vis = self.camshift_track(
                                                    vis, (x00, y00, w, h), hist, hsv, mask)
                                                rects[i] = rec

                                        else:
                                            success, rects = multi_tracker.update(vis)

                                        for i, (x, y, w, h) in enumerate(rects):
                                            p1 = (int(x), int(y))
                                            p2 = (int(x) + int(w), int(y) + int(h))
                                            cv2.rectangle(vis, p1, p2, (255, 0, 0), 2)

                                        cv2.imshow("source", vis)

                            fps.update()
                            fps.stop()
                            cv2.putText(vis, "FPS " + "{:.2f}".format(fps.fps()), (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow('source', vis)

                            ch = 0xFF & cv2.waitKey(1)
                            if ch == 27:
                                break
                        else:
                            break

                    except Exception:
                        continue

                self.source_changed = False
                self.cam.release()
                time.sleep(1)
                self.start_processing()

            else:
                messagebox.showinfo("Could not open webcam or file! ",
                                    "Could not open webcam or file! - Please try again!")

    def detect_haar_cascade(self, vis_roi, cascade_desc):

        gray = cv2.cvtColor(vis_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade = cv2.CascadeClassifier(cascade_desc)
        rects = cascade.detectMultiScale(gray, scaleFactor=self.haar_scale_factor, minNeighbors=self.haar_min_neighbors,
                                         minSize=self.haar_min_size, maxSize=self.haar_max_size, flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) > 0:
            #        rects[:, 2:] += rects[:, :2]
            return rects
        else:
            return None

    def detect_hog_svn(self, vis_roi):

        rects = []
        if vis_roi.shape[1] > 250:

            vis_roi = imutils.resize(vis_roi, width=min(800, vis_roi.shape[1]))
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rects, weights = hog.detectMultiScale(
                vis_roi, winStride=self.winStride, padding=self.roi_padding, scale=self.pyramid_scale, useMeanshiftGrouping=self.meanShift)

        if len(rects) > 0:
            #            rects[:, 2:] += rects[:, :2]
            return rects
        else:
            return None

    def camshift_track(self, vis, rect, hist, hsv, mask):

        x0, y0, w, h = rect

        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        prob &= mask

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     self.max_iter, self.min_movement)
        track_box, track_window = cv2.CamShift(prob, (x0, y0, w, h), term_crit)

        track_window = [track_window[0], track_window[1], track_window[2], track_window[3]]

        return track_window, vis

    def ask_for_tracking_options(self):

        window = Toplevel(self.root)
        window.wm_title("Tracking options")
        window.resizable(width=False, height=False)
        window.geometry('{}x{}'.format(300, 400))
        window.attributes('-topmost', True)

        label1 = Label(window, text="Choose tracking methods?")
        label1.grid(row=0, sticky=W)

        tmp = 0
        for i, (key, value) in enumerate(self.tracking_methods.items()):
            c = Checkbutton(window, text=key, variable=value, onvalue=1, offvalue=0)
            c.grid(row=i + 1, sticky=W)
            tmp += 1

        Label(window, text="CamShift Grouping (0/1):").grid(row=tmp + 1, sticky=W)
        e6 = Entry(window)
        e6.insert(5, self.meanShift)
        e6.grid(row=tmp + 1, column=1)

        Label(window, text="CamShift max_iter:").grid(row=tmp + 2, sticky=W)
        e7 = Entry(window)
        e7.insert(5, self.max_iter)
        e7.grid(row=tmp + 2, column=1)

        Label(window, text="CamShift min_movement:").grid(row=tmp + 3, sticky=W)
        e8 = Entry(window)
        e8.insert(5, self.min_movement)
        e8.grid(row=tmp + 3, column=1)

        btn1 = Button(window, text="Set",
                      command=lambda *args: self.set_tracking_methods(window, e7.get(), e8.get(), e6.get()))
        btn1.grid(row=tmp + 4)

        self.root.wait_window(window)

    def ask_for_detection_options(self):

        window = Toplevel(self.root)
        window.wm_title("Detection options")
        window.resizable(width=False, height=False)
        window.geometry('{}x{}'.format(300, 650))
        window.attributes('-topmost', True)

        label1 = Label(window, text="Choose detection methods?")
        label1.grid(row=0, sticky=W)
        tmp = 0
        for i, (key, value) in enumerate(self.detection_methods.items()):
            c = Checkbutton(window, text=key, variable=value, onvalue=1, offvalue=0)
            c.grid(row=i + 1, sticky=W)
            tmp = i + 1

        Label(window, text="HOG-SVM winStride min:").grid(row=tmp + 1, sticky=W)
        Label(window, text="HOG-SVM winStride max:").grid(row=tmp + 2, sticky=W)
        e1 = Entry(window)
        e2 = Entry(window)
        e1.insert(5, self.winStride[0])
        e2.insert(5, self.winStride[1])
        e1.grid(row=tmp + 1, column=1)
        e2.grid(row=tmp + 2, column=1)

        Label(window, text="HOG-SVM roi_padding min:").grid(row=tmp + 3, sticky=W)
        Label(window, text="HOG-SVM roi_padding max:").grid(row=tmp + 4, sticky=W)
        e3 = Entry(window)
        e4 = Entry(window)
        e3.insert(5, self.roi_padding[0])
        e4.insert(5, self.roi_padding[1])
        e3.grid(row=tmp + 3, column=1)
        e4.grid(row=tmp + 4, column=1)

        Label(window, text="HOG-SVM pyramid_scale (>1.00):").grid(row=tmp + 5, sticky=W)
        e5 = Entry(window)
        e5.insert(5, self.pyramid_scale)
        e5.grid(row=tmp + 5, column=1)

        Label(window, text="Haar Cascade min size x:").grid(row=tmp + 7, sticky=W)
        Label(window, text="Haar Cascade min size y:").grid(row=tmp + 8, sticky=W)
        e7 = Entry(window)
        e8 = Entry(window)
        e7.insert(5, self.haar_min_size[0])
        e8.insert(5, self.haar_min_size[1])
        e7.grid(row=tmp + 7, column=1)
        e8.grid(row=tmp + 8, column=1)

        Label(window, text="Haar Cascade max size x:").grid(row=tmp + 9, sticky=W)
        Label(window, text="Haar Cascade max size y:").grid(row=tmp + 10, sticky=W)
        e11 = Entry(window)
        e12 = Entry(window)
        e11.insert(5, self.haar_max_size[0])
        e12.insert(5, self.haar_max_size[1])
        e11.grid(row=tmp + 9, column=1)
        e12.grid(row=tmp + 10, column=1)

        Label(window, text="Haar Cascade scale_factor (>1.00):").grid(row=tmp + 11, sticky=W)
        e9 = Entry(window)
        e9.insert(5, self.haar_scale_factor)
        e9.grid(row=tmp + 11, column=1)

        Label(window, text="Haar Cascade min_neighbors:").grid(row=tmp + 12, sticky=W)
        e10 = Entry(window)
        e10.insert(5, self.haar_min_neighbors)
        e10.grid(row=tmp + 12, column=1)

        btn1 = Button(window, text="Set",
                      command=lambda *args: self.set_detection_methods(window, e1.get(), e2.get(), e3.get(),
                                                                       e4.get(), e5.get(), e9.get(), e10.get(),
                                                                       e7.get(), e8.get(), e11.get(), e12.get()))
        btn1.grid(row=tmp + 13)

        self.root.wait_window(window)

    def show_masks(self, vis, hsv, lab):

        lower = np.array((float(self.hue_min), float(self.sat_min), float(self.val_min)))
        upper = np.array((float(self.hue_max), float(self.sat_max), float(self.val_max)))

        maskHSV = cv2.inRange(hsv, lower, upper)

        cv2.imshow('source', vis)

        if self.show_mask_image is True:

            cv2.namedWindow('HSV mask', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('HSV mask trackbar', cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("Hue min", 'HSV mask trackbar', self.hue_min, 180, self.set_hue_min)
            cv2.createTrackbar("Hue max", 'HSV mask trackbar', self.hue_max, 180, self.set_hue_max)
            cv2.createTrackbar("Sat. min", 'HSV mask trackbar', self.sat_min, 255, self.set_sat_min)
            cv2.createTrackbar("Sat. max", 'HSV mask trackbar', self.sat_max, 255, self.set_sat_max)
            cv2.createTrackbar("Val. min", 'HSV mask trackbar', self.val_min, 255, self.set_val_min)
            cv2.createTrackbar("Val. max", 'HSV mask trackbar', self.val_max, 255, self.set_val_max)

            if self.show_hsv_mask_conjunction is True:

                frame = cv2.bitwise_and(vis, vis, mask=maskHSV)
                cv2.imshow('HSV mask', frame)

            else:
                cv2.imshow('HSV mask', maskHSV)

        else:
            cv2.destroyWindow('HSV mask')
            cv2.destroyWindow('HSV mask trackbar')

        lowerBGR = np.array((float(self.blue_min), float(self.green_min), float(self.red_min)))
        upperBGR = np.array((float(self.blue_max), float(self.green_max), float(self.red_max)))
        maskBGR = cv2.inRange(vis, lowerBGR, upperBGR)

        if self.show_bgr_mask_image is True:

            cv2.namedWindow('BGR mask', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('BGR mask trackbar', cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("Blue min", 'BGR mask trackbar',
                               self.blue_min, 255, self.set_blue_min)
            cv2.createTrackbar("Blue max", 'BGR mask trackbar',
                               self.blue_max, 255, self.set_blue_max)
            cv2.createTrackbar("Green min", 'BGR mask trackbar',
                               self.green_min, 255, self.set_green_min)
            cv2.createTrackbar("Green max", 'BGR mask trackbar',
                               self.green_max, 255, self.set_green_max)
            cv2.createTrackbar("Red min", 'BGR mask trackbar', self.red_min, 255, self.set_red_min)
            cv2.createTrackbar("Red max", 'BGR mask trackbar', self.red_max, 255, self.set_red_max)

            if self.show_hsv_mask_conjunction is True:

                resultBGR = cv2.bitwise_and(vis, vis, mask=maskBGR)
                cv2.imshow('BGR mask', resultBGR)

            else:
                cv2.imshow('BGR mask', maskBGR)
        else:
            cv2.destroyWindow('BGR mask')
            cv2.destroyWindow('BGR mask trackbar')

        lowerLab = np.array((float(self.lig_min), float(self.gm_min), float(self.by_min)))
        upperLab = np.array((float(self.lig_max), float(self.gm_max), float(self.by_max)))
        maskLab = cv2.inRange(lab, lowerLab, upperLab)

        if self.show_lab_mask_image is True:

            cv2.namedWindow('Lab mask', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Lab mask trackbar', cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("Lightness min", 'Lab mask trackbar',
                               self.lig_min, 255, self.set_lig_min)
            cv2.createTrackbar("Lightness max", 'Lab mask trackbar',
                               self.lig_max, 255, self.set_lig_max)
            cv2.createTrackbar("Green to Magenta min", 'Lab mask trackbar',
                               self.gm_min, 255, self.set_gm_min)
            cv2.createTrackbar("Green to Magenta", 'Lab mask trackbar',
                               self.gm_max, 255, self.set_gm_max)
            cv2.createTrackbar("Blue to Yellow min", 'Lab mask trackbar',
                               self.by_min, 255, self.set_by_min)
            cv2.createTrackbar("Blue to Yellow max", 'Lab mask trackbar',
                               self.by_max, 255, self.set_by_max)

            if self.show_hsv_mask_conjunction is True:

                resultLab = cv2.bitwise_and(vis, vis, mask=maskLab)
                cv2.imshow('Lab mask', resultLab)

            else:
                cv2.imshow('Lab mask', maskLab)
        else:
            cv2.destroyWindow('Lab mask')
            cv2.destroyWindow('Lab mask trackbar')

        if self.mask_to_use == 'HSV mask':
            return maskHSV
        elif self.mask_to_use == 'BGR mask':
            return maskBGR
        else:
            return maskLab

    def show_hists(self, bgr_roi, hsv_roi, lab_roi):

        if self.show_hsv_hist is True:

            hist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            hsv_hist = hist.reshape(-1)
            bin_count = hsv_hist.shape[0]
            bin_w = 24
            img = np.zeros((256, bin_count * bin_w, 3), np.uint8)

            for i in range(bin_count):

                h = int(hsv_hist[i])
                cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                              (int(180.0 * i / bin_count), 255, 255), -1)

            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            cv2.imshow('histo', img)

        else:
            cv2.destroyWindow('histo')

        if self.show_bgr_hist is True:

            if bgr_roi.any():
                b, g, r = cv2.split(bgr_roi)
                b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
                g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
                r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
                hist_w = 512
                hist_h = 400
                bin_w = int(round(hist_w / 256))
                histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
                cv2.normalize(b_hist, b_hist, 0, hist_h, cv2.NORM_MINMAX)
                cv2.normalize(g_hist, g_hist, 0, hist_h, cv2.NORM_MINMAX)
                cv2.normalize(r_hist, r_hist, 0, hist_h, cv2.NORM_MINMAX)

                for i in range(1, 256):
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                             (255, 0, 0), thickness=2)
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(g_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(g_hist[i]))),
                             (0, 255, 0), thickness=2)
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(r_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(r_hist[i]))),
                             (0, 0, 255), thickness=2)

                cv2.imshow('histo_bgr', histImage)

        else:
            cv2.destroyWindow('histo_bgr')

        if self.show_lab_hist is True:

            if lab_roi.any():
                l, a, b = cv2.split(lab_roi)
                l_hist = cv2.calcHist([l], [0], None, [256], [0, 256])
                a_hist = cv2.calcHist([a], [0], None, [256], [0, 256])
                b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
                hist_w = 512
                hist_h = 400
                bin_w = int(round(hist_w / 256))
                histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
                cv2.normalize(l_hist, l_hist, 0, hist_h, cv2.NORM_MINMAX)
                cv2.normalize(a_hist, a_hist, 0, hist_h, cv2.NORM_MINMAX)
                cv2.normalize(b_hist, b_hist, 0, hist_h, cv2.NORM_MINMAX)

                for i in range(1, 256):
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(l_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(l_hist[i]))),
                             (255, 0, 0), thickness=2)
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(a_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(a_hist[i]))),
                             (0, 255, 0), thickness=2)
                    cv2.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                             (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                             (0, 0, 255), thickness=2)
                    cv2.putText(histImage, 'blue=l,green=a,red=b', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                cv2.imshow('histo_lab', histImage)

        else:
            cv2.destroyWindow('histo_lab')

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Do you want to quit?"):

            if self.cam is not None:
                self.cam.release()
            sys.exit()

    def onmouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

        if self.drag_start:
            x0 = min(x, self.drag_start[0])
            y0 = min(y, self.drag_start[1])
            x1 = max(x, self.drag_start[0])
            y1 = max(y, self.drag_start[1])
            self.selection = (x0, y0, x1, y1)

        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None

    def set_scale_factor(self, event=None):
        if event is not None:
            self.scale_factor = float(event.widget.get())

    def set_cam_id(self, event=None):
        if event is not None:
            self.cam_id = int(event.widget.get())

    def set_show_mask(self, event=None):
        print("sss" + str(self.v_mask.get()))
        self.show_mask_image = bool(self.v_mask.get())

    def set_bgr_show_mask(self, event=None):
        self.show_bgr_mask_image = bool(self.v_BGR_mask.get())

    def set_lab_show_mask(self, event=None):
        self.show_lab_mask_image = bool(self.v_Lab_mask.get())

    def set_show_mask_conj(self, event=None):
        self.show_hsv_mask_conjunction = bool(self.v_mask_conj.get())

    def set_show_hsv_hist(self, event=None):
        self.show_hsv_hist = bool(self.v_hsvhist.get())

    def set_show_bgr_hist(self, event=None):
        self.show_bgr_hist = bool(self.v_bgrhist.get())

    def set_show_lab_hist(self, event=None):
        self.show_lab_hist = bool(self.v_labhist.get())

    def set_hue_min(self, val):
        self.hue_min = val

    def set_hue_max(self, val):
        self.hue_max = val

    def set_sat_min(self, val):
        self.sat_min = val

    def set_sat_max(self, val):
        self.sat_max = val

    def set_val_min(self, val):
        self.val_min = val

    def set_val_max(self, val):
        self.val_max = val

    def set_blue_min(self, val):
        self.blue_min = val

    def set_blue_max(self, val):
        self.blue_max = val

    def set_green_min(self, val):
        self.green_min = val

    def set_green_max(self, val):
        self.green_max = val

    def set_red_min(self, val):
        self.red_min = val

    def set_red_max(self, val):
        self.red_max = val

    def set_lig_min(self, val):
        self.lig_min = val

    def set_lig_max(self, val):
        self.lig_max = val

    def set_gm_min(self, val):
        self.gm_min = val

    def set_gm_max(self, val):
        self.gm_max = val

    def set_by_min(self, val):
        self.by_min = val

    def set_by_max(self, val):
        self.by_max = val

    def set_mask(self, event=None):
        if event is not None:
            self.mask_to_use = event.widget.get()

    def set_detection_methods(self, window, win_stride_min, win_stride_max,
                              roi_padding_x, roi_padding_y, pyramid_scale,
                              haar_scale_factor, haar_min_neighbors, haar_min_size_x,
                              haar_min_size_y, haar_max_size_x, haar_max_size_y):

        for key, value in self.detection_methods.items():
            self.detection_methods[key] = BooleanVar(value=value.get())

        self.winStride = (int(win_stride_min), int(win_stride_max))
        self.roi_padding = (int(roi_padding_x), int(roi_padding_y))
        self.pyramid_scale = float(pyramid_scale)
        self.haar_scale_factor = float(haar_scale_factor)
        self.haar_min_neighbors = int(haar_min_neighbors)
        self.haar_min_size = (int(haar_min_size_x), int(haar_min_size_y))
        self.haar_max_size = (int(haar_max_size_x), int(haar_max_size_y))

        window.destroy()

    def set_tracking_methods(self, window, max_iter, min_movement, meanShift):

        for key, value in self.tracking_methods.items():
            self.tracking_methods[key] = BooleanVar(value=value.get())

        self.max_iter = int(max_iter)
        self.min_movement = int(min_movement)
        self.meanShift = int(meanShift)

        window.destroy()


if __name__ == '__main__':
    #    import sys
    App().run()
