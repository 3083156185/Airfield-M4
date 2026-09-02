from abc import abstractmethod

try:   
    import clip
except ImportError as e:
    pass     

import torch  
import torch.nn as nn

from ..utils import smart_inference_mode     
    
class TextModel(nn.Module):
    """Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.
    
    Methods:   
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.  
    """

    def __init__(self):
        """Initialize the TextModel base class."""  
        super().__init__()

    @abstractmethod     
    def tokenize(self, texts):
        """Convert input texts to tokens for model processing.""" 
        pass    

    @abstractmethod   
    def encode_text(self, texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass
 
class MobileCLIPTS(TextModel):
    """Load a TorchScript traced version of MobileCLIP.
 
    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing 
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.    
     
    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder.  
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded. 

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.     
  
    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)   
    """

    def __init__(self, device: torch.device):
        """Initialize the MobileCLIP TorchScript text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.  

        Args:     
            device (torch.device): Device to load the model on.
        """
        super().__init__()  
        from ultralytics.utils.downloads import attempt_download_asset    

        self.encoder = torch.jit.load(attempt_download_asset("mobileclip_blt.ts"), map_location=device)  
        self.tokenizer = clip.clip.tokenize
        self.device = device
   
    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:     
        """Convert input texts to MobileCLIP tokens.
     
        Args:
            texts (list[str]): List of text strings to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
                matching CLIP's behavior to prevent runtime failures on long captions.
   
        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).
   
        Examples:     
            >>> model = MobileCLIPTS("cpu")  
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False   
            ... )  # RuntimeError if exceeds 77-token
        """     
        return self.tokenizer(texts, truncate=truncate).to(self.device)  

    @smart_inference_mode()  
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.
  
        Args:   
            texts (torch.Tensor): Tokenized text inputs.   
            dtype (torch.dtype, optional): Data type for output features.  

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.
  
        Examples:
            >>> model = MobileCLIPTS(device="cpu")    
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens) 
            >>> features.shape 
            torch.Size([2, 512])  # Actual dimension depends on model size     
        """
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)    

class MobileCLIPTS(TextModel):    
    """Load a TorchScript traced version of MobileCLIP.    
   
    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.   

    Attributes:     
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder. 
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.  
  
    Methods:  
        tokenize: Convert input texts to MobileCLIP tokens.    
        encode_text: Encode tokenized texts into normalized feature vectors.   
     
    Examples:    
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        >>> text_encoder = MobileCLIPTS(device=device)  
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])     
        >>> features = text_encoder.encode_text(tokens)
    """
    
    def __init__(self, model_path, device: torch.device):
        """Initialize the MobileCLIP TorchScript text encoder.    
    
        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.   

        Args: 
            device (torch.device): Device to load the model on.
        """
        super().__init__()

        self.encoder = torch.jit.load(model_path, map_location=device)  
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens.
     
        Args:   
            texts (list[str]): List of text strings to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
                matching CLIP's behavior to prevent runtime failures on long captions.   

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).
    
        Examples:  
            >>> model = MobileCLIPTS("cpu") 
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(  
            ...     ["a very long caption"], truncate=False 
            ... )  # RuntimeError if exceeds 77-token    
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)     

    @smart_inference_mode()    
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.
 
        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.    

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:  
            >>> model = MobileCLIPTS(device="cpu")  
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"]) 
            >>> features = model.encode_text(tokens)
            >>> features.shape  
            torch.Size([2, 512])  # Actual dimension depends on model size    
        """
        # NOTE: no need to do normalization here as it's embedded in the torchscript model   
        return self.encoder(texts).to(dtype) 

class MobileCLIP2TS(TextModel):  
    """Load a TorchScript traced version of MobileCLIP2.
  
    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing   
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.

    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder.
        tokenizer (callable): Tokenizer function for processing text inputs.  
        device (torch.device): Device where the model is loaded.     
   
    Methods: 
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples: 
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)  
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"]) 
        >>> features = text_encoder.encode_text(tokens)  
    """   
 
    def __init__(self, model_path, device: torch.device):
        """Initialize the MobileCLIP TorchScript text encoder.    
     
        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.

        Args:
            device (torch.device): Device to load the model on.
        """   
        super().__init__()
    
        self.encoder = torch.jit.load(model_path, map_location=device)
        self.tokenizer = clip.clip.tokenize 
        self.device = device 

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:   
        """Convert input texts to MobileCLIP tokens.    

        Args:
            texts (list[str]): List of text strings to tokenize. 
            truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
                matching CLIP's behavior to prevent runtime failures on long captions. 
     
        Returns:     
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).    
  
        Examples:
            >>> model = MobileCLIPTS("cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"]) 
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # RuntimeError if exceeds 77-token
        """     
        return self.tokenizer(texts, truncate=truncate).to(self.device)
   
    @smart_inference_mode()     
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor: 
        """Encode tokenized texts into normalized feature vectors.    

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.    
     
        Returns: 
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.     

        Examples:
            >>> model = MobileCLIPTS(device="cpu")  
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)    
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """    
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)   
