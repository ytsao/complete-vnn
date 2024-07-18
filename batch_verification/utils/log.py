class Logger:
    log_file_name: str = "log.txt"

    @staticmethod
    def debugging(meesages: str) -> None:
        with open(Logger.log_file_name, "+a") as f:
            f.write(f"[DEBUGGING]: {meesages}\n")

        return


    @staticmethod
    def info(messages: str) -> None:
        with open(Logger.log_file_name, "+a") as f:
            f.write(f"[INFO]{messages}\n")

        return 


    @staticmethod
    def error(messages: str) -> None: 
        with open(Logger.log_file_name, "+a") as f:
            f.write(f"[ERROR]{messages}\n")

        return 



