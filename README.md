# Video Annotation Project

## Overview

This project performs automatic person annotation and tracking in a video using YOLO and OpenCV.

## Features

- Detects persons in video frames
- Automatically tracks moving persons
- Draws bounding boxes around detected persons
- Assigns tracking IDs
- Bounding boxes automatically change according to the person's movement
- Saves the annotated video

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- YOLO11
- Computer Vision

## Project Structure

```text
Video_Annotation_Project/
│
├── annotate_video.py
├── README.md
├── .gitignore
├── frames/
├── labels/
├── output/
└── video/
