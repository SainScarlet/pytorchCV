from openvino.inference_engine import IECore

ie = IECore()
for device in ie.available_devices:
    print(device)
