#!/usr/bin/env python
# coding: utf-8

from tkinter import *
import torch
from torchvision import transforms
import face_recognition
import cv2
import time
import numpy as np
import os
from PIL import ImageTk, Image

LOADED = False
BASE_PATH = os.path.abspath(os.path.dirname(__file__))


def authentification():
    """
    Fonction qui permet de s'identifier lors du lancement de la Jetson
    """

    # Load a sample picture and learn how to recognize it.
    maxime_image = face_recognition.load_image_file("images/maxime.jpeg")
    maxime_face_encoding = face_recognition.face_encodings(maxime_image)[0]
 
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        maxime_face_encoding,
    ]
    known_face_names = [
		"Maxime"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    authorized = False
    video_capture = cv2.VideoCapture(0)
    SEUIL = 50
    lst_cpt = []
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            code = cv2.COLOR_BGR2RGB
            rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    if name in known_face_names:
                        lst_cpt.append(name)
                    if len(lst_cpt) == SEUIL:
                        if len(set(lst_cpt)) == 1:
                            authorized = True
                            video_capture.release()
                            cv2.destroyAllWindows()
                            return authorized
                        else:
                            lst_cpt = []
                    face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # Display the resulting image
        cv2.imshow("Video", frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return authorized


def fire_detection():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fire_model = torch.load("models/MobileNetV3_20230425171303.pt").to(device)
    # fire_model = torch.load('models/MobileNet_20230422201227.pt').to(device)
    # fire_model = torch.load('models/MobileNetV3_20230427170313-2.pt').to(device)
    fire_model.train(False)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]
    )
    classes = ["Fire", "No fire", "Start fire"]
    #
    videos_to_test = ["firetest.mp4", "fire5.mp4", "fire4.mp4"]
    for video_to_test in videos_to_test:
        cam_port = os.path.join(BASE_PATH, "tests/" + video_to_test)
        cam = cv2.VideoCapture(cam_port)
        #
        prev_frame_time = 0
        new_frame_time = 0
        #
        while cam.isOpened():
            success, img = cam.read()
            if not success:
                break
            data = transform(img).to(device)
            data.unsqueeze_(0)
            klass = fire_model(data).cpu().detach().numpy()

            # notification
            pred_class = classes[np.argmax(klass)]
            # tester si feu oou d2but de feu
            # if......

            cv2.putText(
                img,
                classes[np.argmax(klass)],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                thickness=2,
                lineType=2,
            )
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.imshow("Fire Detection", cv2.resize(img, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.release()
    cv2.destroyAllWindows()
    #
    del fire_model




def main():
    # Etape 1: authentification pour accéder à l'interface graphique
    """
    authorized = authentification()
    if not authorized:
        print("No authorization")
        from sys import exit
        exit(0)
    """

    # Etape 2: interface graphique
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("1000x650")
    root.title("Edge AI System")
    
    # Main frame
    main_frame = Frame(root, relief=RIDGE, borderwidth=2)
    main_frame.config(background="blue1")
    main_frame.pack(fill=BOTH, expand=1)
    #
    # Welcome message for user
    label_msg = Label(
        main_frame, text=("Welcome!"), bg="blue1", font=("Helvetica 24 bold"), height=2
    )
    label_msg.pack(side=TOP)
    label_msg2 = Label(
        main_frame,
        text=("Hello, you are well authorized, congrats !"),
        bg="blue1",
        font=("Helvetica 22 bold"),
    )
    label_msg2.pack(side=TOP)
    
    # add logos to the interface
    logo1 = Image.open("logos/UMONS-EN-rvb.png")
    logo1 = logo1.resize((140, 80), Image.LANCZOS) 
    logo1 = ImageTk.PhotoImage(logo1)
    logo_label1 = Label(main_frame, image=logo1)
    logo_label1.image = logo1
    logo_label1.place(x=50, y=550)


    logo2 = Image.open("logos/fpms.png")
    logo2 = logo2.resize((140, 80), Image.LANCZOS)
    logo2 = ImageTk.PhotoImage(logo2)
    logo_label2 = Label(main_frame, image=logo2)
    logo_label2.image = logo2
    logo_label2.place(x=200, y=550)

    logo3 = Image.open("logos/numediart.png")
    logo3 = logo3.resize((140, 80), Image.LANCZOS)
    logo3 = ImageTk.PhotoImage(logo3)
    logo_label3 = Label(main_frame, image=logo3)
    logo_label3.image = logo3
    logo_label3.place(x=350, y=550)

    
    logo4 = Image.open("logos/deepilia.png")
    logo4 = logo4.resize((140, 80), Image.LANCZOS)
    logo4 = ImageTk.PhotoImage(logo4)
    logo_label4 = Label(main_frame, image=logo4)
    logo_label4.image = logo4
    logo_label4.place(x=500, y=550)

    logo5 = Image.open("logos/infortech.png")
    logo5 = logo5.resize((140, 80), Image.LANCZOS)
    logo5 = ImageTk.PhotoImage(logo5)
    logo_label4 = Label(main_frame, image=logo5)
    logo_label4.image = logo5
    logo_label4.place(x=650, y=550)

    logo6 = Image.open("logos/LOGO_F114_Mohammed-BENJELLOUN.png")
    logo6 = logo6.resize((140, 80), Image.LANCZOS)
    logo6 = ImageTk.PhotoImage(logo6)
    logo_label4 = Label(main_frame, image=logo6)
    logo_label4.image = logo6
    logo_label4.place(x=800, y=550)

    
    # Ajout texte
    label_msg3 = Label(
        main_frame,
        text=("Version initiale"),
        bg="blue2",
        font=("Helvetica 20 bold"),
    )
    label_msg3.place(x=150, y=120)
    label_msg4 = Label(
        main_frame,
        text=("Version compressée"),
        bg="blue2",
        font=("Helvetica 20 bold"),
    )
    label_msg4.place(x=550, y=120)
    # Menu
    but1 = Button(
        main_frame,
        padx=5,
        pady=5,
        width=25,
        height=2,
        bg="white",
        fg="black",
        relief=RAISED,
        command=fire_detection,
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but2 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=fire_detection,
        text="XAI",
        font=("helvetica 16 bold"),
    )

    but3 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=fire_detection,
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but4 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=fire_detection,
        text="XAI",
        font=("helvetica 16 bold"),
    )
    but1.place(x=150, y=200)
    but2.place(x=150, y=300)
    but3.place(x=550, y=200)
    but4.place(x=550, y=300)


    but5 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=15,
        bg="white",
        fg="black",
        relief=RAISED,
        command=root.destroy,
        text="Exit",
        font=("helvetica 15 bold"),
    )
    but5.place(x=670, y=440)

    root.mainloop()


if __name__ == "__main__":
    main()
