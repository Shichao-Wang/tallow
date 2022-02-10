from tallow.trainers.hooks import early_stop

# FILE_NAME = "e1_b469.earlystop.pt"
FILE_NAME = "e1_b469_m0.9021000266075134.earlystop.pt"

result = early_stop.MODEL_FILE_RE.match(FILE_NAME)
print(result.groupdict())
