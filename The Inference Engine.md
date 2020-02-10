# The Inference Engine

:star: The Inference Engine provides a library of computer vision functions and performs the inference on a model.

The Inference Engine **runs the actual inference** on a model. It only works with the **Intermediate
Representations** that come from the Model Optimizer, or the Intel® Pre-Trained Models in OpenVINO™
that are already in IR format.<br>
Where the **Model Optimizer** made some **improvements to size and complexity of the models**
to **improve memory and computation times**, the **Inference Engine** provides **hardware-based 
optimizations** to get even further improvements from a model. This really empowers your
application to run at the edge and use up as little of device resources as possible.<br>
The Inference Engine has a straightforward API to allow easy integration into your edge application.
The **Inference Engine itself is actually built in C++** (at least for the CPU version), leading to
overall faster operations, However, it is very common to utilize the **built-in Python wrapper** to 
interact with it in Python code.

**Developer Documentation**: https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html

-----------

### Supported Devices

The supported devices for the Inference Engine are **all Intel® hardware**, and are a variety of 
such devices: **CPUs, including integrated graphics processors, GPUs, FPGAs, and VPUs.**<br>
FPGAs, or Field Programmable Gate Arrays, are able to be further configured by a customer after
manufacturing. Hence the “field programmable” part of the name.<br>
VPUs, or **Vision Processing Units**, are going to be like the Intel® Neural Compute Stick.
They are small, but powerful devices that can be plugged into other hardware, for the 
specific purpose of accelerating computer vision tasks.

**Developer Documentation**: https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html

------------

### Using the Inference Engine with an IR

To load an IR into the Inference Engine, 2 main classes in `openvino.inference_engine` library is mostly used while working 
in Python:
* `IECore`: which is a Python wrapper to work with the Inference Engine.
* `IENetwork`: which is what is intially hold the network and get loaded into the `IECore`.
In order to use `IECore`, no arguments are needed to initialize. But in `IENetwork`, one need to load arguments named `model`
and `weights` to initialize - the XML files and Binary Files that make up the the model's Intermediate Representations.

#### Check Supported Layers

In the IECore documentation, there was another function called `query_network`,
which takes in an **IENetwork as an argument and a device name**, 
and **returns a list of layers the Inference Engine supports**.
One can then iterate through the layers in the IENetwork created, and check whether
they are in the supported layers list. If a layer was not supported, a CPU extension may be able to help.<br>
The `device_name` argument is just a string for which device is being used -
`”CPU”`, `”GPU”`, `”FPGA”`, or `”MYRIAD”` (which applies for the Neural Compute Stick).

#### CPU extension

If layers were successfully built into an Intermediate Representation with the Model Optimizer,
some may still be unsupported by default with the Inference Engine when run on a CPU. However, 
there is likely support for them using one of the available CPU extensions.<br>
These do differ by operating system a bit, although they should still be in the
same overall location. If you navigate to your OpenVINO™ install directory,
then `deployment_tools`, `inference_engine`, `lib`, `intel64`:
* On Linux, few CPU extension files available for AVX and SSE.
  * Intel® Atom processors use SSE4, while Intel® Core processors will utilize AVX.
  * This is especially important to make note of when transferring a program from a 
    Core-based laptop to an Atom-based edge device. If the incorrect extension is specified 
    in the application, the program will crash.
  * AVX systems can run SSE4 libraries, but not vice-versa.
* On Mac, there’s just a single CPU extension file.
One can add these directly to the `IECore` using their full path. 
After adding the CPU extension, if necessary, one should re-check that all layers are now supported.
If they are, it’s finally time to load the model into the IECore.

------------

### Sending Inference Requests to the IE

After loading `IENetwork` into the `IECore`, you get back an `ExecutableNetwork`, which is what you will send inference 
requests to. The are 2 type of inference requests one can make:
* Synchronous
* Asynchronous

With an `ExecutableNetwork`, synchronous requests just use the `infer` function, while asynchronous requests begin with 
`start_async`, and then one can wait until request is complete. These requests are `InferRequest` object, which will hold
both the output and input of the request.

------------
### Synchronous Vs Asynchronous

|             Example                         |       Async Vs Sync               |
|:--------------------------------------------|----------------------------------:|
|A network call is made to a server with an unknown latency for returning a response, and the user is otherwise able to use the app while waiting on a response.  |       Asynchronous                |
|The application needs to wait for a user input before being able to process additional data.           |       Synchronous                 |

------------

### Handling Results
The Inference Requests are stored in a `requests` attribute in the `ExecutableNetwork`.`InferRequest` object having a `wait` function, means they are asynchronous requests.<br>
Each `InferRequest` also has a few attributes - named `inputs`, `outputs`, and `latency`. Inputs can be an image frame, ouptuts contains the results and latency notes the inference time of the current request.<br>
It may be useful to print out the outputs to know what they contain after a request is complete.For this, the `data` under the `“prob”` key, or sometimes `output_blob` (see related documentation), to get **an array of the probabilities returned from the inference request.**

**Developer Documentation for Blob Class Inference**: https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Blob.html

------------

## :star2: Behind The Scenes In the Inference Engine
The Inference Engine is built and optimized in C++, although that’s just the CPU version. There are some differences in what is actually occurring under the hood with the different devices. You are able to work with a shared API to interact with the Inference Engine, while largely being able to ignore these differences.<br>
#### WHY C++?
Many different Computer Vision and AI frameworks are built with C++, and have additional Python interfaces. OpenCV and TensorFlow, for example, are built primarily in C++, but many users interact with the libraries in Python. C++ is faster and more efficient than Python when well implemented, and it also gives the user more direct access to the items in memory and such, and they can be passed between modules more efficiently.<br>
C++ is compiled & optimized ahead of runtime, whereas Python basically gets read line by line when a script is run. On the flip side, Python can make it easier for prototyping and fast fixes. It’s fairly common then to be using a C++ library for the actual Computer Vision techniques and inferencing, but with the application itself in Python, and interacting with the C++ library via a Python API.

#### Optimizations By Device
The exact optimizations differ by device with the Inference Engine. While from your end interacting with the Inference Engine is mostly the same, there’s actually separate plugins within for working with each device type.<br>
**CPUs**, for instance, rely on the **Intel® Math Kernel Library for Deep Neural Networks**, or MKL-DNN. CPUs also have some extra work to help improve device throughput, especially for CPUs with higher numbers of cores.<br>
**GPUs** utilize the **Compute Library for Deep Neural Networks**, or clDNN, which uses OpenCL within. Using OpenCL introduces a small overhead right when the GPU Plugin is loaded, but is only a one-time overhead cost. The GPU Plugin works best with FP16 models over FP32 models<br>.
Getting to VPU devices, like the Intel® Neural Compute Stick, there are additional costs associated with it being a USB device. It’s actually recommended to be processing four inference requests at any given time, in order to hide the costs of data transfer from the main device to the VPU.

------------

### Further References:

* [Model Optimizer with raspberry pi](https://software.intel.com/en-us/articles/model-downloader-optimizer-for-openvino-on-raspberry-pi)
* [Integrating the Inference Engine with the application](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_with_customer_application_new_API.html)
* [Object Detection Demo SSD Async](https://github.com/opencv/open_model_zoo/blob/master/demos/object_detection_demo_ssd_async/README.md)
* [Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html) for more on differences in optimizations between devices.

