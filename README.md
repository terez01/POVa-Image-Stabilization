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
Examples:
`python3 main.py --video <path_to_video> --optical_flow_method farneback`

`python3 main.py --video <path_to_input_video> --model searaft_models/Tartan-C-T-TSKH-spring540x960-M.pth --device cuda --output <path_to_output> --optical_flow_method sea-raft`

### Sources
- SEA-RAFT: https://github.com/princeton-vl/SEA-RAFT
- optical flow visualization: https://github.com/tomrunia/OpticalFlow_Visualization