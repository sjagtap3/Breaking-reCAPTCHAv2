# No CAP(TCHA) - Enhancing CAPTCHA Solvers and Using Adversarial Examples to Make CAPTCHA More Robust

## Overview
In our study, we build on previous work that focuses on breaking Google’s reCAPTCHA v2 using advanced machine learning-based approaches, specifically object detection. The prior research demonstrates a method that can solve 100% of presented CAPTCHA challenges. However, we have identified several directions for extending this work, which we define as the primary focus of our project.

## Goals
The goals of our study include:

1. **Improving Accuracy and Confidence**
   - Update the object detection model to enhance overall prediction accuracy and confidence.

2. **Enhancing Performance on Type 2 reCAPTCHA v2 Challenges**
   - Fine-tune the existing image segmentation model on a new dataset to improve performance on Type 2 reCAPTCHA v2 challenges.

3. **Exploring Adversarial Examples**
   - Attempt to break the solver using adversarial examples presented in other literature.

## Usage
- Run `solve_recaptcha.py` to solve a single reCAPTCHA challenge:
  ```
  python solve_recaptcha.py
  ```

- Run `test_environment.py` to solve multiple reCAPTCHA challenges and create a log file:
  ```
  python test_environment.py
  ```

## Project Structure
This project has the following directory structure:

- `assets/`: Contains the images used in the README.
- `IP/`: Contains the script for changing IP address.
  - `vpn.py`: Script for changing IP address (only works on macOS).
- `models/`: Contains all the models used in the project.
  - `YOLO_Classification/`: Contains the YOLO model for classification.
  - `YOLO_Segment/`: Contains the YOLO model for segmentation.
- `utils/`: Contains utility scripts.
  - `collect_data.py`: Script for collecting data for the classification model.
  - `label_tool.py`: A UI tool for labeling CAPTCHA images.
  - `visualize_log_files.py`: Analyzes log files, providing insights into task attempts, their distribution, and key statistical measures.
- `README.md`: This file, providing an overview of the project.
- `solve_recaptcha.py`: Demo script for solving a single reCAPTCHA challenge using Selenium.
- `test_environment.py`: Demo script for solving multiple reCAPTCHA challenges using Selenium and creating a log file.
- `requirements.txt`: Contains the required libraries for the project.


