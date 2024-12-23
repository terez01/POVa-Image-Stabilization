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
Simple example:
`python3 main.py --video <path_to_input_video> --method raft`

Visualize optical flow and analyze stabilization example:
`python3 main.py --video <path_to_input_video> --output <path_to_output_video> --method raft --visualize_flow --analyze_stabilization`

Save optical flow example:
`python3 main.py --video <path_to_input_video> --save_flow <path_to_flow_output> --method raft`

Load optical flow example:
`python3 main.py --video <path_to_input_video> --load_flow <path_to_input_flow> --method raft`

---

### Avaliable optical flow methods
- Farneback (`--method farneback`): running on CPU
- DeepFlow (`--method deepflow`): running on CPU
- RAFT (`--method raft`): running on GPU (if avaliable)

---

### Sources
- https://arxiv.org/abs/2003.12039
- https://pytorch.org/vision/main/models/raft.html
- https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html

---

### Datasets used
- Datasets with ground truths:
    - https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
    - http://sintel.is.tue.mpg.de/downloads
- Datasets without ground truths:
    - https://github.com/cxjyxxme/deep-online-video-stabilization?tab=readme-ov-file
    - https://github.com/jiy173/selfievideostabilization