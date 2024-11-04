# No CAP(TCHA)
# Enhancing CAPTCHA Solvers and Using Adversarial Examples to Make CAPTCHA More Robust

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

## Expected Outcomes
We hope that our extensions will yield significant performance gains and valuable insights related to the security and robustness of reCAPTCHA v2, benefiting future research efforts.


| CAPTCHA Example | CAPTCHA Result |
|:---:|:---:|
| <img src="assets/type2_example.gif" width="200" /> | <img src="assets/type2_example_result.png" width="200" /> |

## Requirements
- Python 3.9
- Firefox (Geckodriver)
- Required libraries (see `requirements.txt`)

## Installation
1. Clone the repository

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Download and set up Geckodriver:
   - Download Geckodriver from the official website: [Geckodriver Releases](https://github.com/mozilla/geckodriver/releases)
   - Extract the downloaded archive and add the path to the `geckodriver` executable to your system's PATH environment variable.

## Usage
- Run `solve_recaptcha.py` to solve a single reCAPTCHA challenge:
  ```
  python solve_recaptcha.py
  ```

- Run `test_environment.py` to solve multiple reCAPTCHA challenges and create a log file:
  ```
  python test_environment.py
  ```

## Data
- The training data for the classification task can be found [here](https://drive.google.com/drive/folders/19kET6PFXHaHZqzr9DU_ZsgX-n13Ef4sj?usp=sharing).
- The validation data for the classification task can be found [here](https://drive.google.com/drive/folders/19kG2NQls2iH1sUq0js0MOArBiJiOmVGM?usp=sharing).

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

## System Compatibility
Please note that the code in this project was developed and tested on macOS M1. While most of the code should be compatible with other operating systems, certain functionalities, such as the VPN script (`vpn.py`), may not work on non-macOS systems. If you encounter any issues running the code on a different operating system, please refer to the documentation of the specific libraries or tools used in the project for guidance on how to set them up for your system.

