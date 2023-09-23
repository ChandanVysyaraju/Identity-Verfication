import os

# Set the path to the extracted LFW dataset
dataset_path = "C:\\Users\\chand\\Desktop\\CV_Project\\dataset\\lfw"

# Get the list of people in the dataset
people = os.listdir(dataset_path)

# Create a dictionary to store labels for each person
labels = {}
label_counter = 0

# Open the train.txt file to write image paths and labels
with open("train.txt", "w") as f:
    for person in people:
        # Get the list of images for the current person
        person_images = os.listdir(os.path.join(dataset_path, person))

        # Assign a label to the current person if not already assigned
        if person not in labels:
            labels[person] = label_counter
            label_counter += 1

        # Write the image paths and labels to the train.txt file
        for image in person_images:
            img_path = os.path.join(dataset_path, person, image)
            f.write(f"{img_path} {labels[person]}\n")
