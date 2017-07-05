import os
import csv
import shutil

cwd = os.getcwd()
dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
image_id = 0
new_csv = []

print("Will combine the data from these folders: {}".format(dirs))

for directory in dirs:
    if "sorted" in directory:
        continue

    print("Working on {}.".format(directory))

    # Open the CSV in each directory
    with open(directory + "/frame_data.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if "steering" in row:
                continue

            image, motor, steer = row[0], row[1], row[2]
            # print("Image: {} Motor: {} Steering: {}".format(image, motor, steer))

            shutil.copy(directory + "/" + image, "sorted/" + str(image_id) + ".jpg")
            frame = {"jpg": image_id, "steering": steer, "throttle": motor}
            new_csv.append(frame)
            image_id += 1

    with open(cwd + "/sorted/frame_data.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["jpg", "steering", "throttle"])
        for frame in new_csv:
            csvwriter.writerow([frame["jpg"], frame["steering"], frame["throttle"]])

print("Saving new csv in the 'sorted' folder...")

print("Saved csv in {}.".format(cwd + "/sorted/frame_data.csv"))
print("Done.")