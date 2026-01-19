class Logger:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(module, message):
        print(f"{Logger.BLUE}[{module}]{Logger.ENDC} {message}")

    @staticmethod
    def success(module, message):
        print(f"{Logger.GREEN}[{module}]{Logger.ENDC} {message}")

    @staticmethod
    def warning(module, message):
        print(f"{Logger.WARNING}[{module} - WARN]{Logger.ENDC} {message}")

    @staticmethod
    def error(module, message):
        print(f"{Logger.FAIL}[{module} - ERROR]{Logger.ENDC} {message}")
    
    @staticmethod
    def graph(action, detail):
        print(f"{Logger.CYAN}[Graph]{Logger.ENDC} {Logger.BOLD}{action}{Logger.ENDC}: {detail}")
        
    @staticmethod
    def nav(action, detail):
        print(f"{Logger.HEADER}[Nav]{Logger.ENDC} {Logger.BOLD}{action}{Logger.ENDC}: {detail}")

    @staticmethod
    def json(module, data, label="Data"):
        import json
        formatted = json.dumps(data, indent=2)
        print(f"{Logger.BLUE}[{module}]{Logger.ENDC} {label}:\n{formatted}")
