# inference/deployment.py
import torch
import torch.nn as nn
import numpy as np
import time
import onnx
import os
from typing import List, Dict, Tuple, Optional, Union, Any

class DeploymentManager:
    """
    Deployment manager for voice authentication models
    
    This class handles optimization and deployment of models for inference,
    including ONNX export and TensorRT integration.
    """
    def __init__(
        self,
        auth_model: nn.Module,
        antispoofing_model: nn.Module,
        deployment_dir: str = 'deployment'
    ):
        self.auth_model = auth_model
        self.antispoofing_model = antispoofing_model
        self.deployment_dir = deployment_dir
        
        # Create deployment directory
        os.makedirs(deployment_dir, exist_ok=True)
    
    def export_onnx(
        self,
        sample_size: Tuple[int, int, int] = (1, 1, 16000),
        quantize: bool = True,
        optimize: bool = True
    ):
        """
        Export models to ONNX format
        
        Args:
            sample_size: Input sample size (batch, channels, time)
            quantize: Whether to quantize the model
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Paths to exported models
        """
        # Ensure PyTorch and ONNX are properly imported
        import torch.onnx
        
        try:
            import onnxruntime
            import onnxoptimizer
            has_onnx_optimizer = True
        except ImportError:
            print("ONNX optimizer not available. Install with: pip install onnxoptimizer")
            has_onnx_optimizer = False
            optimize = False
        
        # Paths for exported models
        auth_path = os.path.join(self.deployment_dir, 'authentication.onnx')
        antispoofing_path = os.path.join(self.deployment_dir, 'antispoofing.onnx')
        
        # Create dummy input
        dummy_input = torch.randn(sample_size)
        
        # Export authentication model
        print("Exporting authentication model to ONNX...")
        self.auth_model.eval()
        
        torch.onnx.export(
            self.auth_model,
            dummy_input,
            auth_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'embedding'],
            dynamic_axes={
                'input': {2: 'time'},
                'output': {0: 'batch'},
                'embedding': {0: 'batch'}
            }
        )
        
        # Export anti-spoofing model
        print("Exporting anti-spoofing model to ONNX...")
        self.antispoofing_model.eval()
        
        torch.onnx.export(
            self.antispoofing_model,
            dummy_input,
            antispoofing_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'fusion_weights', 'individual_outputs', 'embeddings'],
            dynamic_axes={
                'input': {2: 'time'},
                'output': {0: 'batch'}
            }
        )
        
        # Optimize ONNX models if requested
        if optimize and has_onnx_optimizer:
            print("Optimizing ONNX models...")
            
            # Optimize authentication model
            auth_model_onnx = onnx.load(auth_path)
            auth_model_optimized = onnxoptimizer.optimize(auth_model_onnx)
            onnx.save(auth_model_optimized, auth_path)
            
            # Optimize anti-spoofing model
            antispoofing_model_onnx = onnx.load(antispoofing_path)
            antispoofing_model_optimized = onnxoptimizer.optimize(antispoofing_model_onnx)
            onnx.save(antispoofing_model_optimized, antispoofing_path)
        
        # Quantize ONNX models if requested
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                print("Quantizing ONNX models...")
                
                # Quantize authentication model
                auth_path_quantized = os.path.join(self.deployment_dir, 'authentication_quantized.onnx')
                quantize_dynamic(
                    auth_path,
                    auth_path_quantized,
                    weight_type=QuantType.QUInt8
                )
                
                # Quantize anti-spoofing model
                antispoofing_path_quantized = os.path.join(self.deployment_dir, 'antispoofing_quantized.onnx')
                quantize_dynamic(
                    antispoofing_path,
                    antispoofing_path_quantized,
                    weight_type=QuantType.QUInt8
                )
                
                # Update paths to quantized models
                auth_path = auth_path_quantized
                antispoofing_path = antispoofing_path_quantized
                
            except ImportError:
                print("ONNX quantization not available. Install with: pip install onnxruntime-tools")
        
        print(f"ONNX export complete. Models saved to {self.deployment_dir}")
        
        return auth_path, antispoofing_path
    
    # inference/deployment.py (continued)
    def create_tensorrt_engines(
        self,
        onnx_model_paths: List[str],
        precision: str = 'fp16'
    ):
        """
        Create TensorRT engines from ONNX models
        
        Args:
            onnx_model_paths: Paths to ONNX models
            precision: Precision for TensorRT ('fp32', 'fp16', or 'int8')
            
        Returns:
            Paths to TensorRT engine files
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("TensorRT or PyCUDA not available. Install TensorRT and run:")
            print("pip install pycuda")
            return None
        
        engine_paths = []
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        for onnx_path in onnx_model_paths:
            # Create engine path
            engine_path = onnx_path.replace('.onnx', '.engine')
            
            # Build TensorRT engine
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:
                
                # Configure builder
                config = builder.create_builder_config()
                
                # Set precision
                if precision == 'fp16' and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                elif precision == 'int8' and builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    # For INT8, you would need to provide a calibration dataset
                    # This is a simplified version without calibration
                
                # Set maximum workspace size (4GB)
                config.max_workspace_size = 4 << 30
                
                # Parse ONNX model
                with open(onnx_path, 'rb') as model:
                    if not parser.parse(model.read()):
                        print(f"ERROR: Failed to parse ONNX model {onnx_path}")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        continue
                
                # Build and save engine
                print(f"Building TensorRT engine for {onnx_path}...")
                engine = builder.build_engine(network, config)
                
                if engine:
                    with open(engine_path, 'wb') as f:
                        f.write(engine.serialize())
                    print(f"TensorRT engine saved to {engine_path}")
                    engine_paths.append(engine_path)
                else:
                    print(f"Failed to create TensorRT engine for {onnx_path}")
        
        return engine_paths
    
    def optimize_for_mobile(
        self,
        target_platform: str = 'android'
    ):
        """
        Optimize models for mobile deployment
        
        Args:
            target_platform: Target mobile platform ('android' or 'ios')
            
        Returns:
            Paths to optimized model files
        """
        try:
            import torch.utils.mobile_optimizer as mobile_optimizer
        except ImportError:
            print("Mobile optimizer not available in this PyTorch version")
            return None
        
        # Paths for exported models
        auth_path = os.path.join(self.deployment_dir, f'authentication_mobile_{target_platform}.pt')
        antispoofing_path = os.path.join(self.deployment_dir, f'antispoofing_mobile_{target_platform}.pt')
        
        # Prepare models for mobile
        print(f"Optimizing models for {target_platform}...")
        
        # Optimize authentication model
        self.auth_model.eval()
        
        # Trace the model
        example_input = torch.rand(1, 1, 16000)
        traced_auth_model = torch.jit.trace(self.auth_model, example_input)
        
        # Optimize for mobile
        optimized_auth_model = mobile_optimizer.optimize_for_mobile(traced_auth_model)
        
        # Save the model
        optimized_auth_model.save(auth_path)
        
        # Optimize anti-spoofing model
        self.antispoofing_model.eval()
        
        # Trace the model
        traced_antispoofing_model = torch.jit.trace(self.antispoofing_model, example_input)
        
        # Optimize for mobile
        optimized_antispoofing_model = mobile_optimizer.optimize_for_mobile(traced_antispoofing_model)
        
        # Save the model
        optimized_antispoofing_model.save(antispoofing_path)
        
        print(f"Mobile optimization complete. Models saved to {self.deployment_dir}")
        
        return auth_path, antispoofing_path
