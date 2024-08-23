import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the Branch Net
class BranchNet(nn.Module):
    def __init__(self,dim=2):
        super(BranchNet, self).__init__()
        if dim == 1:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*64*64, 256)
        self.fc2 = nn.Linear(256, 128)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)
        return z

# Define the Trunk Net
class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
    
    def forward(self, xy):
        xy = torch.relu(self.fc1(xy))
        xy = torch.relu(self.fc2(xy))
        return self.fc3(xy)

# Define the DeepONet model
class DeepONet(nn.Module):
    def __init__(self,dim=2):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(dim)
        self.trunk_net = TrunkNet()
        self.dim = dim
        if dim == 1:
            self.fc_out1 = nn.Linear(128, 1)
        else:
            self.fc_out1 = nn.Linear(128, 1)
            self.fc_out2 = nn.Linear(128, 1)
    
    def forward(self, field, xy):
        z = self.branch_net(field)
        xy = self.trunk_net(xy)
        if self.dim == 1:
            output1 = self.fc_out1(z * xy)
            return output1
        else:
            output1 = self.fc_out1(z * xy)
            output2 = self.fc_out2(z * xy)
            return torch.cat([output1, output2], dim=1)
        
    def save(self,path):
        torch.save(self.trunk_net.state_dict(), path+"/trunk_net.pt")
        torch.save(self.branch_net.state_dict(), path+"/branch_net.pt")
    
    def load(self,path):
        self.trunk_net.load_state_dict(torch.load(path+"/trunk_net.pt"))
        self.branch_net.load_state_dict(torch.load(path+"/branch_net.pt"))
        # torch.save(self.trunk_net.state_dict(), path+"/trunk_net.pt")
        # torch.save(self.branch_net.state_dict(), path+"/branch_net.pt")

# Generate data for demonstration
def generate_data(num_samples):

    a = torch.rand(())*5
    b = torch.rand(())*5
    c = torch.rand(())*5
    d = torch.rand(())*5

    x = y = torch.linspace(-1,1,64)
    yy_t,xx_t = torch.meshgrid(x,y)
    # xx_t,yy_t = torch.Tensor(xx),torch.Tensor(yy)

    def vec_field_lyap_fn(x,y):
        f1 = y
        f2 = -a*x - 2*torch.tanh(b*x) - c*y - 2*torch.tanh(d*y)
        V = a/2 * x**2 + 2*torch.log(torch.cosh(b*x))/b + y**2/2
        return f1,f2,V
    
    f1,f2,V = vec_field_lyap_fn(xx_t,yy_t)
    # f1 = yy_t
    # f2 = -a*xx_t - 2*torch.tanh(b*xx_t) - c*yy_t - 2*torch.tanh(d*yy_t)
    # V = a/2 * xx_t**2 + 2*torch.log(torch.cosh(b*xx_t))/b + yy_t**2/2

    m = f2.abs().max()
    if m > 1:
        f1 = f1 / m
        f2 = f2 / m

    mv = V.abs().max()
    V = V / mv

    field_f = torch.stack((f1,f2)).unsqueeze(0)
    field_v = V.unsqueeze(0).unsqueeze(0)

    coordinates = torch.rand(num_samples, 2) * 2 - 1  # (batch_size, 2) in [-1,1]^2

    coordinates_x = coordinates[...,0]
    coordinates_y = coordinates[...,1]

    target_f1,target_f2,target_V = vec_field_lyap_fn(coordinates_x,coordinates_y)
    # target_f1 = coordinates_y
    # target_f2 = -a*coordinates_x - 20*torch.tanh(b*coordinates_x) - c*coordinates_y - 20*torch.tanh(d*coordinates_y)
    # target_V = a/2 * coordinates_x**2 + 2*torch.log(torch.cosh(b*coordinates_x))/b + coordinates_y**2/2

    target_f = torch.stack((target_f1,target_f2),dim=1)

    if m > 1:
        target_f = target_f / m
    target_V = target_V / mv
    
    return field_f, field_v, coordinates, target_f, target_V

# Training loop
def train_model(vector_field_model, lyapunov_model, criterion, optimizer, epochs=100):
    vector_field_model.train()
    lyapunov_model.train()
    field_f, field_v, coordinates, target_f, target_V = generate_data(1000)
    for _ in range(epochs):
        # running_loss = 0.0
        # for i, (field_f, field_v, coordinates, target_f, target_V) in enumerate(train_loader):
        optimizer.zero_grad()
        output_f = vector_field_model(field_f, coordinates)
        # loss = criterion(output, target)
        output_v = lyapunov_model(field_v, coordinates)
        loss = criterion(output_f, target_f) + criterion(output_v, target_V)
        loss.backward()
        optimizer.step()
    running_loss = loss.item()
    return running_loss
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Main execution
if __name__ == "__main__":
    epochs = 1000000 // 20
    num_samples = 1000

    # Initialize model, loss function, and optimizer
    vector_field_model = DeepONet()
    lyapunov_model = DeepONet(1)

    vector_field_model.load("trained_models/vector_field")
    lyapunov_model.load("trained_models/lyapunov")

    # print("Loaded weights")
    log_file = "training_log.txt"  # Specify the log file path

    with open(log_file, "a") as f:  # Open the log file in append mode
        f.write(f"Loaded weights\n")

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(vector_field_model.parameters()) + list(lyapunov_model.parameters()), lr=0.001)
    
    for epoch in range(epochs):
        # Generate synthetic data
        # field_f, field_v, coordinates, target_f, target_V = generate_data(num_samples)
        
        # Create DataLoader
        # dataset = TensorDataset(field_f, field_v, coordinates, target_f, target_V)
        # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
        # Train the model
        running_loss = train_model(vector_field_model, lyapunov_model, criterion, optimizer, epochs=20)

        with open(log_file, "a") as f:  # Open the log file in append mode
            f.write(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}\n")

        # if epoch % 100 == 0:
        vector_field_model.save("trained_models/vector_field")
        lyapunov_model.save("trained_models/lyapunov")
