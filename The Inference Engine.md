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


------------

### Further References:

* [Model Optimizer with raspberry pi](https://software.intel.com/en-us/articles/model-downloader-optimizer-for-openvino-on-raspberry-pi)
* [Integrating the Inference Engine with the application](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_with_customer_application_new_API.html)
* [Object Detection Demo SSD Async](https://github.com/opencv/open_model_zoo/blob/master/demos/object_detection_demo_ssd_async/README.md)
