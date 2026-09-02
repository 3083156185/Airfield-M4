import os, contextlib, platform, re, cv2   
import importlib.metadata
import matplotlib.pyplot as plt
import numpy as np   
from pathlib import Path
from tqdm import tqdm as tqdm_original
from contextlib import contextmanager
    
import torch

RANK = int(os.getenv("RANK", -1))
VERBOSE = str(os.getenv("DEIM_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans   

def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""     
    return string.encode().decode("ascii", "ignore") if WINDOWS else string   
  
class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.  
 
    Examples:   
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)    
        >>> def func(): 
        >>> # Function logic here
        >>>     pass
 
        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here    
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg 
        self.verbose = verbose     
     
    def __enter__(self):  
        """Executes when entering TryExcept context, initializes instance."""   
        pass
  
    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""    
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))     
        return True

def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.
  
    Example: 
        decorator: @plt_settings({"font.size": 12})    
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """   
    if rcparams is None:
        rcparams = {"font.size": 11}   

    def decorator(func):   
        """Decorator to apply temporary rc parameters and backend to a function."""
    
        def wrapper(*args, **kwargs):     
            """Sets rc parameters and backend, calls the original function, and restores the settings."""    
            original_backend = plt.get_backend() 
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8   
                plt.switch_backend(backend)     
     
            # Plot with backend and always revert to original backend     
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:   
                    plt.close("all") 
                    plt.switch_backend(original_backend)  
            return result
     
        return wrapper  
    
    return decorator

def increment_path(path, exist_ok=False, sep="", mkdir=False):   
    """  
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.  

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to 
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a 
    directory if it does not already exist.   

    Args:
        path (str | pathlib.Path): Path to increment.  
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        sep (str): Separator to use between the path and the incrementation number.
        mkdir (bool): Create a directory if it does not exist.

    Returns:  
        (pathlib.Path): Incremented path.    

    Examples:
        Increment a directory path:     
        >>> from pathlib import Path     
        >>> path = Path("runs/exp")    
        >>> new_path = increment_path(path)    
        >>> print(new_path)
        runs/exp2
   
        Increment a file path:   
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)  
        >>> print(new_path)
        runs/exp/results2.txt
    """   
    path = Path(path)  # os-agnostic     
    if path.exists() and not exist_ok:  
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1  
        for n in range(2, 9999):     
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break     
        path = Path(p) 
 
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
  
class TQDM(tqdm_original):     
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.
  
    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options.  
    
    Attributes:    
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not 
            explicitly set.

    Methods:     
        __init__: Initializes the TQDM object with custom settings.

    Examples: 
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    """

    def __init__(self, *args, **kwargs):
        """   
        Initializes a custom TQDM progress bar.    

        This class extends the original tqdm class to provide customized behavior for Ultralytics projects.

        Args: 
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.     

        Notes:  
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.    

        Examples:
            >>> from ultralytics.utils import TQDM     
            >>> for i in TQDM(range(100)):
            ...     # Your code here     
            ...     pass 
        """    
        # kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # logical 'and' with default value if passed 
        # kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)  
   
def parse_version(version="0.0.0") -> tuple:
    """   
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version.
   
    Args:     
        version (str): Version string, i.e. '2.0.1+cpu'
  
    Returns:     
        (tuple): Tuple of integers representing the numeric part of the version, i.e. (2, 0, 1)    
    """    
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1) 
    except Exception as e: 
        print(f"failure for parse_version({version}), returning (0, 0, 0): {e}")  
        return 0, 0, 0    

def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",    
    name: str = "version",    
    hard: bool = False, 
    verbose: bool = False,  
    msg: str = "",
) -> bool:  
    """
    Check current version against the required version or range.     
    
    Args:   
        current (str): Current version or package name to get version from.   
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.   
        verbose (bool): If True, print warning message if requirement is not met.     
        msg (str): Extra message to display if verbose.
    
    Returns: 
        (bool): True if requirement is met, False otherwise. 

    Examples:
        Check if current version is exactly 22.04   
        >>> check_version(current="22.04", required="==22.04")  

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04 
        >>> check_version(current="22.04", required="<=22.04") 

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")    
    """    
    if not current:  # if current is '' or None
        print(f"invalid check_version({current}, {required}) requested, please check values.")    
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg     
            current = metadata.version(current)  # get version string from package name  
        except metadata.PackageNotFoundError as e: 
            if hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False 

    if not required:  # if required is '' or None  
        return True   
  
    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)    
        or (MACOS and "macos" not in required and "darwin" not in required)    
    ):
        return True

    op = ""
    version = ""
    result = True   
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op: 
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)  
        if op == "==" and c != v:     
            result = False    
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False  
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):    
            result = False   
        elif op == "<" and not (c < v):
            result = False
    if not result:   
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met 
        if verbose:
            print(warning)  
    return result   

def read_device_model() -> str:
    """  
    Read the device model information from the system and cache it for quick access.
   
    Returns:
        (str): Kernel release information.
    """
    return platform.release().lower()  

def is_jetson(jetpack=None) -> bool:    
    """     
    Determine if the Python environment is running on an NVIDIA Jetson device.  
    
    Args: 
        jetpack (int | None): If specified, check for specific JetPack version (4, 5, 6).   

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    if jetson := ("tegra" in DEVICE_MODEL):     
        if jetpack:  
            try:     
                content = open("/etc/nv_tegra_release").read()  
                version_map = {4: "R32", 5: "R35", 6: "R36"}  # JetPack to L4T major version mapping
                return jetpack in version_map and version_map[jetpack] in content   
            except Exception:   
                return False
    return jetson
   
DEVICE_MODEL = read_device_model()
IS_JETSON = is_jetson()

@contextmanager   
def arange_patch(args):
    """
    Workaround for ONNX torch.arange incompatibility with FP16. 
     
    https://github.com/pytorch/pytorch/issues/148041.   
    """
    if args.dynamic and args.half and args.format == "onnx":  
        func = torch.arange
   
        def arange(*args, dtype=None, **kwargs):
            """Return a 1-D tensor of size with values from the interval and common difference."""     
            return func(*args, **kwargs).to(dtype)  # cast to dtype instead of passing dtype  
     
        torch.arange = arange  # patch
        yield
        torch.arange = func  # unpatch
    else:
        yield     

def smart_inference_mode():  
    """Apply torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator.""" 
     
    def decorate(fn):   
        """Apply appropriate torch decorator for inference mode based on torch version."""  
        if torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through    
        else:  
            return (torch.inference_mode)()(fn) 
    
    return decorate 

def visualize_masks(masks, classes, class_names, title, scores=None):
    """
    将多个mask合并可视化，不同类别用不同颜色   
    """     
    if len(masks) == 0:    
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 获取mask尺寸    
    h, w = masks[0].shape  
    
    # 创建彩色图像
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)    
   
    # 为每个类别分配颜色 
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)     
    
    # 叠加每个mask
    for i, (mask, cls) in enumerate(zip(masks, classes)):
        color = colors[int(cls) % len(colors)]
        mask_bool = mask > 0     
        vis_img[mask_bool] = color * 0.6 + vis_img[mask_bool] * 0.4
 
    return vis_img
   
def create_comparison_plot(save_dir, label_masks, label_cls, pred_masks, pred_cls, pred_scores, 
                          class_names, image_id, iou_threshold=0.2, max_display=15):
    """
    创建GT和Prediction的对比图，按IoU排序
    """ 
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))   
 
    # Ground Truth
    if len(label_masks) > 0: 
        gt_vis = visualize_masks(label_masks, label_cls, class_names, "Ground Truth")   
        axes[0].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Ground Truth ({len(label_masks)} masks)', fontsize=14, fontweight='bold')     
        axes[0].axis('off') 
        
        # 统计每个类别的数量
        from collections import Counter
        cls_counter = Counter([class_names[int(cls)] for cls in label_cls])  
        gt_text = "GT Classes:\n" + "\n".join([f"  {cls_name}: {count}"  
                                                for cls_name, count in sorted(cls_counter.items())])
        
        axes[0].text(0.02, 0.98, gt_text, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.5))
    else:
        axes[0].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', 
                    fontsize=16, color='red')
        axes[0].axis('off')   
    
    # Predictions 
    if len(pred_masks) > 0 and len(label_masks) > 0:  
        # 计算每个预测mask与所有GT mask的IoU  
        pred_ious = [] 
        for pred_mask in pred_masks:
            max_iou = 0    
            for gt_mask in label_masks:    
                iou = calculate_mask_iou(pred_mask, gt_mask)  
                max_iou = max(max_iou, iou) 
            pred_ious.append(max_iou) 
        
        pred_ious = np.array(pred_ious)
        
        # 过滤低IoU的预测
        high_iou_indices = [i for i, iou in enumerate(pred_ious) if iou >= iou_threshold]
    
        if len(high_iou_indices) > 0: 
            filtered_pred_masks = pred_masks[high_iou_indices]   
            filtered_pred_cls = pred_cls[high_iou_indices]
            filtered_pred_ious = pred_ious[high_iou_indices]     
    
            pred_vis = visualize_masks(filtered_pred_masks, filtered_pred_cls, class_names, "Predictions")   
            axes[1].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB)) 
            axes[1].set_title(f'Predictions ({len(filtered_pred_masks)}/{len(pred_masks)} masks, IoU≥{iou_threshold})',   
                            fontsize=14, fontweight='bold')
            axes[1].axis('off')
   
            # 按IoU排序并分组显示
            pred_info = list(zip(filtered_pred_cls, filtered_pred_ious)) 
            pred_info_sorted = sorted(pred_info, key=lambda x: x[1], reverse=True)   
    
            # 限制显示数量，避免太多
            if len(pred_info_sorted) > max_display:     
                display_info = pred_info_sorted[:max_display]
                pred_text = f"Top {max_display} Predictions (by IoU):\n"   
            else:    
                display_info = pred_info_sorted  
                pred_text = "Predictions (sorted by IoU):\n" 
    
            pred_text += "\n".join([f"  {class_names[int(cls)]}: IoU={iou:.3f}" 
                                   for cls, iou in display_info])
            
            if len(pred_info_sorted) > max_display:
                pred_text += f"\n  ... and {len(pred_info_sorted) - max_display} more"
  
            # 添加统计信息  
            pred_text += f"\n\nIoU Stats:"
            pred_text += f"\n  Mean: {np.mean(filtered_pred_ious):.3f}"     
            pred_text += f"\n  Max: {np.max(filtered_pred_ious):.3f}"
            pred_text += f"\n  Min: {np.min(filtered_pred_ious):.3f}"  
    
            axes[1].text(0.02, 0.98, pred_text, transform=axes[1].transAxes, 
                        fontsize=9, verticalalignment='top', family='monospace',   
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.5))  
        else:
            axes[1].text(0.5, 0.5, f'No Predictions with IoU≥{iou_threshold}\n({len(pred_masks)} total predictions)', 
                        ha='center', va='center', fontsize=14, color='orange')
            axes[1].axis('off')   
    elif len(pred_masks) > 0 and len(label_masks) == 0:
        # 没有GT，显示所有预测
        pred_vis = visualize_masks(pred_masks, pred_cls, class_names, "Predictions")     
        axes[1].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB))   
        axes[1].set_title(f'Predictions ({len(pred_masks)} masks, No GT for IoU)',    
                        fontsize=14, fontweight='bold')  
        axes[1].axis('off') 
    
        from collections import Counter     
        pred_counter = Counter([class_names[int(cls)] for cls in pred_cls]) 
        pred_text = "Pred Classes (No GT):\n" + "\n".join([f"  {cls_name}: {count}" 
                                                for cls_name, count in sorted(pred_counter.items())])  
        axes[1].text(0.02, 0.98, pred_text, transform=axes[1].transAxes,  
                    fontsize=10, verticalalignment='top', family='monospace', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.5))   
    else:  
        axes[1].text(0.5, 0.5, 'No Predictions', ha='center', va='center',     
                    fontsize=16, color='red')  
        axes[1].axis('off') 
    
    plt.suptitle(f'Image ID: {image_id}', fontsize=16, fontweight='bold')   
    plt.tight_layout()  
    plt.savefig(str(save_dir / "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def calculate_mask_iou(mask1, mask2):  
    """
    计算两个mask的IoU
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()    
    
    if union == 0:    
        return 0.0
   
    return intersection / union  
