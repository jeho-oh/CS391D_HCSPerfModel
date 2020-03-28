
log_fh = None

def openlog(logfile=None):
    global log_fh
    if logfile is None:
        logfile = "log.txt"
    try:
        log_fh = open(logfile,'w')
    except IOError as e:
        sys.exit("ERROR: Could not open log file {}. Quitting. Err message: {}".format(logfile,e))

def closelog():
    log_fh.close()

def logmsg(msg,vars=None,tab=0,log_only=False):
    if vars is not None:
        msg = tab * "    " + msg.format(*vars)
    if not log_only:
        print(str(msg))
    global log_fh
    try:
        log_fh.write(str(msg) + "\n")
    except AttributeError as e:
        openlog("log.txt")
        try:
            log_fh.write(msg + "\n")
        except AttributeError as e:
            sys.exit("Could not write to the log file. Err message: {}".format(e))