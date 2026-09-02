import sys, logging, torch  
  
class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {   
        "TIME": "\033[92m",       # 绿色
        "LOCATION": "\033[93m",   # 黄色   
        "INFO": "\033[37m",       # 白色  
        "DEBUG": "\033[36m",      # 青色     
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色   
        "CRITICAL": "\033[1;31m", # 红色加粗
    }

    def format(self, record):
        time_str = f"{self.COLORS['TIME']}{self.formatTime(record, self.datefmt)}{self.RESET}"   
        location = f"{self.COLORS['LOCATION']}[{record.filename}:{record.funcName}:{record.lineno}]{self.RESET}"     
        level_color = self.COLORS.get(record.levelname, self.RESET)  
        levelname = f"{level_color}{record.levelname}{self.RESET}"
        message = f"{level_color}{record.getMessage()}{self.RESET}"

        return f"{time_str} {location} {levelname}: {message}"    
 
def is_dist_available_and_initialized():   
    if not torch.distributed.is_available(): 
        return False
    if not torch.distributed.is_initialized():
        return False   
    return True  

def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank() 
  
class DistributedLogger(logging.Logger):
    """支持分布式的 Logger，每次输出时自动检查"""
    
    def __init__(self, name, level=logging.INFO):  
        super().__init__(name, level)
        self._base_level = level    
        self._last_rank = None    
        self._ensure_handlers()
        if not is_dist_available_and_initialized():   
            self.is_dist_initialized = False    
 
    def _ensure_handlers(self):    
        """确保 handlers 配置正确"""
        self.is_dist_initialized = is_dist_available_and_initialized()     
        current_rank = get_rank() if self.is_dist_initialized else -1
        
        # 如果 rank 变了，重新配置  
        if self._last_rank != current_rank:
            self.handlers.clear()     
            
            is_main_process = (not self.is_dist_initialized) or (current_rank == 0)
   
            if is_main_process:     
                ch = logging.StreamHandler(sys.stdout)   
                ch.setLevel(self._base_level) 
                formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")  
                ch.setFormatter(formatter)    
                self.addHandler(ch)
                self.setLevel(self._base_level)    
            else:    
                self.addHandler(logging.NullHandler())
                self.setLevel(logging.CRITICAL)     
   
            self.propagate = False
            self._last_rank = current_rank     
    
    def _log(self, level, msg, args, **kwargs):     
        """重写 _log 方法，每次输出前检查""" 
        if not self.is_dist_initialized:    
            self._ensure_handlers()     
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        super()._log(level, msg, args, **kwargs)     

def get_logger(name=None, level=logging.INFO):
    """    
    创建并返回一个分布式安全的 logger
    """
    # 设置自定义 logger 类 
    old_class = logging.getLoggerClass()
    logging.setLoggerClass(DistributedLogger)  
  
    logger = logging.getLogger(name)   
    if isinstance(logger, DistributedLogger):     
        logger._base_level = level 
        logger._ensure_handlers()  
    
    logging.setLoggerClass(old_class)
   
    return logger     
 
def test_logger():
    logger = get_logger(__name__)
    logger.info("info")
    logger.warning("info")     
    logger.error("info") 
