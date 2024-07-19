class Logger:
    log_file_name: str = ""
    with_log_file: bool = True

    @staticmethod
    def initialize(filename: str = "log.txt", with_log_file: bool = True) -> None:
        Logger.log_file_name = filename
        Logger.with_log_file = with_log_file


    @staticmethod
    def debugging(messages: str) -> None:
        if Logger.with_log_file:
            with open(Logger.log_file_name, "+a") as f:
                f.write(f"[DEBUGGING]: {messages}\n")
        
        print("\033[93m {}\033[00m".format(f"[DEBUGGING] {messages}"))
        
        return


    @staticmethod
    def info(messages: str) -> None:
        if Logger.with_log_file:
            with open(Logger.log_file_name, "+a") as f:
                f.write(f"[INFO]{messages}\n")
        
        print("\033[92m {}\033[00m".format(f"[INFO] {messages}"))
        
        return 


    @staticmethod
    def error(messages: str) -> None: 
        if Logger.with_log_file:
            with open(Logger.log_file_name, "+a") as f:
                f.write(f"[ERROR]{messages}\n")
        
        print("\033[91m {}\033[00m".format(f"[ERROR] {messages}"))
        
        return 



