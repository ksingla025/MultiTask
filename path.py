#!/usr/bin/python

import commands

DATA = "./data/"
DATA_BI = DATA+"parallel/"
DATA_MONO = DATA+"mono/"

DATA_TASK = DATA+"task/"
DATA_PROCESSED = DATA +"processed/"
DATA_MONO_CLEAN = DATA_PROCESSED + "mono_clean/"
commands.getstatusoutput("mkdir -p " + DATA_MONO_CLEAN)

DATA_BI_CLEAN = DATA_PROCESSED + "bi_clean/"
commands.getstatusoutput("mkdir -p " + DATA_BI_CLEAN)

DATA_ID = DATA_PROCESSED + "word2id/"
commands.getstatusoutput("mkdir -p " + DATA_ID)

DATA_BATCH = DATA_PROCESSED + "batch/"
commands.getstatusoutput("mkdir -p " + DATA_BATCH)

LOGS_PATH = './logs_multi/'
commands.getstatusoutput("mkdir -p " + LOGS_PATH)