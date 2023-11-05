import torch

"""remap the id of device for .pth file"""

# Model weight file path
model_path = '/home/sunho/Documents/mygit/TrackAndRe-ID/weights/reid/myweights/model1106.imgtri.pth.tar-800'

# Set the device for model loading
device = torch.device('cuda:0')

# Load the model (You should either import the model class or define it)
# For example, if the model class is MyModel:
# model = MyModel()

# Load the model weight file and change the device
model = torch.load(model_path, map_location=device)

# Now the model is allocated on device 0

# Save the model to disk (including its state dictionary)
# Set a new file path
new_model_path = '/home/sunho/Documents/mygit/TrackAndRe-ID/weights/reid/myweights/model1106.imgtri.cuda0.pth'

# Save the entire model
torch.save(model, new_model_path)