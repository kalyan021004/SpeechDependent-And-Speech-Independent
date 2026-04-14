from recognize_speaker import recognize
import os

test_file1 = "test/yweweler.wav"

test_file2= "test/theo.wav"
test_file3 = "test/nicolas.wav"
test_file4= "test/lucas.wav"
test_file5 = "test/jackson.wav"

# Check if file exists
if not os.path.exists(test_file2):
    print("Error: test.wav not found in main folder")
else:
    speaker = recognize(test_file2)
    print("Detected Speaker:", speaker)