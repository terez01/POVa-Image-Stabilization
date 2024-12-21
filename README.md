# POVa-Image-Stabilization
An application compensating for unwanted camera motion in video capture.

---

### Setup
Run the following command to create a virtual environment:
`python3 -m venv venv`

Activate the virtual environment:
`source venv/bin/activate`

Install the dependencies listed in the requirements file:
`pip install -r requirements.txt`

To update the requirements file run:
`pip freeze > requirements.txt`

---

### Run the application
Example:
`python3 main.py --video <path_to_input_video> --output <path_to_output> --method raft --visualize_flow`

### Sources
- https://arxiv.org/abs/2003.12039
- https://pytorch.org/vision/main/models/raft.html
- https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html

### Datasets
- https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
- http://sintel.is.tue.mpg.de/downloads 