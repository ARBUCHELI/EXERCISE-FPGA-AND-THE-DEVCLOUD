# EXERCISE-FPGA-AND-THE-DEVCLOUD

## Disclaimer: This exercise was Created by Udacity (AWS Machine Learning Foundations Course) (Solution of the exercise and adaptation as a repository: Andrés R. Bücheli.)

Requesting an edge node with a CPU and Intel® Arria 10 FPGA on Intel's DevCloud, loading a model on the Intel® Arria 10 FPGA, and running inference on an image.

In this exercise, you will do the following:

1. Write a Python script to load a model and run inference 10 times on a device on Intel's DevCloud.
    * Calculate the time it takes to load the model.
    * Calculate the time it takes to run inference 10 times.
2. Write a shell script to submit a job to Intel's DevCloud.
3. Submit a job using <code>qsub</code> on an <strong>IEI Tank-870</strong> edge node with an <strong>Intel® Arria 10 FPGA</strong>.
4. Run <code>liveQStat</code> to view the status of your submitted jobs.
5. Retrieve the results from your job.
6. View the results.

<strong>IMPORTANT: Set up paths so we can run Dev Cloud utilities</strong>
You must run this every time you enter a Workspace session.

<pre><code>
%env PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))
</code></pre>

# The Model
We will be using the code><vehicle-license-plate-detection-barrier-0106</code> model for this exercise. Remember that to run a model on the FPGA, we need to use 
<code>FP16</code> as the model precision.

The model has already been downloaded for you in the <code>/data/models/intel</code> directory on Intel's DevCloud.

We will be running inference on an image of a car. The path to the image is code></data/resources/car.png</code>

# Step 1: Creating a Python Script
The first step is to create a Python script that you can use to load the model and perform inference. We'll use the <code>%%writefile</code> magic to create a Python file
called <code>inference_on_device.py</code>. In the next cell, you will need to complete the <code>TODO</code> items for this Python script.

<code>TODO</code> items:

1. Load the model
2. Get the name of the input node
3. Prepare the model for inference (create an input dictionary)
4. Run inference 10 times in a loop

<pre><code>
%%writefile inference_on_device.py

import time
import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    model=IENetwork(model_structure, model_weights)

    core = IECore()
    net = core.load_network(network=model, device_name=args.device, num_requests=1)
    print(f"Time taken to load model = {time.time()-start} seconds")

    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)

    # Running Inference in a loop on the same image
    input_dict={input_name:input_img}

    start=time.time()
    for _ in range(10):
        net.infer(input_dict)
    
    print(f"Time Taken to run 10 Inference on FPGA is = {time.time()-start} seconds")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    
    args=parser.parse_args() 
    main(args)
</code></pre>

# Step 2: Creating a Job Submission Script
To submit a job to the DevCloud, you'll need to create a shell script. Similar to the Python script above, we'll use the <code>%%writefile</code> magic command to create a 
shell script called <code>inference_fpga_model_job.sh</code>. In the next cell, you will need to complete the <code>TODO</code> items for this shell script.

<code>TODO</code> items:

1. Create three variables:
    * <code>DEVICE</code> - Assign the value as the first argument passed into the shell script.
    * <code>MODELPATH</code> - Assign the value as the second argument passed into the shell script.
2. Call the Python script using the three variable values as the command line argument

<pre><code>
%%writefile inference_fpga_model_job.sh
#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

DEVICE=$1
MODELPATH=$2


source /opt/intel/init_openvino.sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP16_MobileNet_Clamp.aocx


# Run the load model python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE}

cd /output

tar zcvf output.tgz stdout.log stderr.log
</code></pre>

# Step 3: Submitting a Job to Intel's DevCloud
In the next cell, you will write your <code>!qsub</code> command to load your model and run inference on the <strong>IEI Tank-870</strong> edge node with an 
<strong>Intel Core i5</strong> CPU and an <code>Intel® Arria 10 FPGA</code>.

Your <code>!qsub</code> command should take the following flags and arguments:

1. The first argument should be the shell script filename
2. <code>-d</code> flag - This argument should be <code>.</code>
3. <code>-l</code> flag - This argument should request an edge node with an <strong>IEI Tank-870</strong>. The default quantity is 1, so the 1 after <code>nodes</code> is optional.
    * <strong>Intel Core i5 6500TE</strong> for your <code>CPU</code>.
    * <strong>Intel® Arria 10</strong> for your <code>FPGA</code>.
4. <code>-F</code> flag - This argument should contain the two values to assign to the variables of the shell script:

    * <strong>DEVICE</strong> - Device type for the job: <code>FPGA</code>. Remember that we need to use the <strong>Heterogenous plugin</strong> (HETERO) to run inference on the FPGA.
    * <strong>MODELPATH</strong> - Full path to the model for the job. As a reminder, the model is located in <code>/data/models/intel</code>.
    
<strong>Note</strong>: There is an optional flag, <code>-N</code>, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to name your job submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.

<pre><code>
job_id_core = !qsub inference_fpga_model_job.sh -d . -l nodes=1:tank-870:i5-6500te:iei-mustang-f100-a10 -F "HETERO:FPGA,CPU /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106" -N store_core 
print(job_id_core[0])
</code></pre>

# Step 4: Running liveQStat
Running the <code>liveQStat</code> function, we can see the live status of our job. Running the this function will lock the cell and poll the job status 10 times. The cell is locked until this finishes polling 10 times or you can interrupt the kernel to stop it by pressing the stop button at the top:

  * <code>Q</code> status means our job is currently awaiting an available node
  * <code>R</code> status means our job is currently running on the requested node
  
<strong>Note</strong>: In the demonstration, it is pointed out that W status means your job is done. This is no longer accurate. Once a job has finished running, it will no longer show in the list when running the <code>liveQStat</code> function.

<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

# Step 5: Retrieving Output Files
In this step, we'll be using the <code>getResults</code> function to retrieve our job's results. This function takes a few arguments.

1. <code>job id</code> - This value is stored in the <code>job_id_core</code> variable we created during <strong>Step 3</strong>. Remember that this value is an array with a single string, so we access the string value using <code>job_id_core[0]</code>.
2. <code>filename</code> - This value should match the filename of the compressed file we have in our <code>inference_fpga_model_job.sh shell script</code>.
3. <code>blocking</code> - This is an optional argument and is set to <code>False</code> by default. If this is set to <code>True</code>, the cell is locked while waiting for the results to come back. There is a status indicator showing the cell is waiting on results.

<strong>Note</strong>: The <code>getResults</code> function is unique to Udacity's workspace integration with Intel's DevCloud. When working on Intel's DevCloud environment, your job's results are automatically retrieved and placed in your working directory.

<pre><code>
import get_results

get_results.getResults(job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<code>!tar zxf output.tgz</code>
<code>!cat stdout.log</code>
<code>!cat stderr.log</code>

# Solution of the exercise and adaptation as a Repository: Andrés R. Bücheli.






































