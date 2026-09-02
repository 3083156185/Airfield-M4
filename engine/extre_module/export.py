import torch, json, logging
from pathlib import Path
from .utils import IS_JETSON    

def export_onnx(
    torch_model,
    im,
    onnx_file,
    opset,
    input_names,
    output_names,
    dynamic,   
) -> None: 
    """
    Export a PyTorch model to ONNX format.

    Args:     
        torch_model (torch.nn.Module): The PyTorch model to export.     
        im (torch.Tensor): Example input tensor for the model.     
        onnx_file (str): Path to save the exported ONNX file. 
        opset (int): ONNX opset version to use for export. 
        input_names (list[str]): List of input tensor names. 
        output_names (list[str]): List of output tensor names.
        dynamic (bool | dict, optional): Whether to enable dynamic axes.  

    Notes:  
        Setting `do_constant_folding=True` may cause issues with DNN inference for torch>=1.12.
    """ 
    torch.onnx.export(
        torch_model,    
        im,
        onnx_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False  
        input_names=input_names,   
        output_names=output_names,
        dynamic_axes=dynamic or None, 
    )     

def export_engine(
    onnx_file,    
    engine_file,   
    workspace,     
    half,
    dynamic,
    shape,     
    dla,     
    metadata=None,  
    verbose=False,   
    prefix= "",
) -> None: 
    """
    Export a YOLO model to TensorRT engine format.    
  
    Args:     
        onnx_file (str): Path to the ONNX file to be converted.  
        engine_file (str, optional): Path to save the generated TensorRT engine file.
        workspace (int, optional): Workspace size in GB for TensorRT.  
        half (bool, optional): Enable FP16 precision. 
        dynamic (bool, optional): Enable dynamic input shapes.
        shape (tuple[int, int, int, int], optional): Input shape (batch, channels, height, width). 
        dla (int, optional): DLA core to use (Jetson devices only).
        metadata (dict, optional): Metadata to include in the engine file. 
        verbose (bool, optional): Enable verbose logging.   
        prefix (str, optional): Prefix for log messages.
   
    Raises:     
        ValueError: If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: If the ONNX file cannot be parsed.  
    
    """
    import tensorrt as trt  # noqa
  
    engine_file = engine_file or Path(onnx_file).with_suffix(".engine")

    logger = trt.Logger(trt.Logger.INFO) 
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    # Engine builder    
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10  # is TensorRT >= 10  
    if is_trt10 and workspace > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)  
    elif workspace > 0:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace 
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)   
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half
 
    # Optionally switch to DLA if enabled
    if dla is not None:     
        if not IS_JETSON:
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        print(f"{prefix} enabling DLA on core {dla}...")
        if not half:     
            raise ValueError(
                "DLA requires either 'half=True' (FP16) to be enabled. Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA    
        config.DLA_core = int(dla)   
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)   

    # Read ONNX file
    parser = trt.OnnxParser(network, logger)  
    if not parser.parse_from_file(onnx_file):  
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")    
     
    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]     
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:  
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')   
 
    if dynamic:   
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # minimum input shape  
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # max input shape
        for inp in inputs:
            if inp.name == 'images':    
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
   
    print(f"{prefix} building {'FP' + ('16' if half else '32')} engine as {engine_file}") 
    if half: 
        config.set_flag(trt.BuilderFlag.FP16)     
 
    # Write file
    build = builder.build_serialized_network if is_trt10 else builder.build_engine  
    with build(network, config) as engine, open(engine_file, "wb") as t:
        # Metadata 
        if metadata is not None:   
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))    
            t.write(meta.encode())
        # Model
        t.write(engine if is_trt10 else engine.serialize())     
