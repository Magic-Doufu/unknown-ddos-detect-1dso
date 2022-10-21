import torch
import torch.nn as nn
import torch.nn.functional as F

class DHRNet(nn.Module):

    def __init__(self, num_classes, latent = 2):
        super(DHRNet,self).__init__()
        self.num_classes = num_classes

        self.conv1_1 =  nn.Conv1d(1, 16, kernel_size=3,
                               stride=1, padding=1)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 =  nn.Conv1d(16, 16, kernel_size=3,
                               stride=1, padding=1)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 =  nn.Conv1d(16, 32, kernel_size=3,
                               stride=1, padding=1)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 =  nn.Conv1d(32, 32, kernel_size=3,
                               stride=1, padding=1)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 =  nn.Conv1d(32, 32, kernel_size=3,
                               stride=1, padding=1)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 =  nn.Conv1d(32, 32, kernel_size=3,
                               stride=1, padding=1)
        self.prelu3_2 = nn.PReLU()
        self.conv3_3 =  nn.Conv1d(32, 4, kernel_size=1,
                               stride=1, padding=0)
        self.prelu3_3 = nn.PReLU()


        self.btl1 = nn.Conv1d(16, 1, kernel_size=3,
                               stride=1, padding=1)
        self.prelubtl1 = nn.PReLU()
        self.btlu1 = nn.Conv1d(1, 16, kernel_size=3,
                               stride=1, padding=1)
        self.btl2 = nn.Conv1d(32, 8, kernel_size=3,
                               stride=1, padding=1)
        self.prelubtl2 = nn.PReLU()
        self.btlu2 = nn.Conv1d(8, 32, kernel_size=3,
                               stride=1, padding=1)
        self.btl3 = nn.Conv1d(32, 8, kernel_size=3,
                               stride=1, padding=1)
        self.prelubtl3 = nn.PReLU()
        self.btlu3 = nn.Conv1d(8, 32, kernel_size=3,
                               stride=1, padding=1)

        self.fc4 = nn.Linear(4 * 70, latent)
        self.prelufc4 = nn.PReLU()
        self.fc5 = nn.Linear(latent, self.num_classes)

        self.deconv1 = nn.ConvTranspose1d(16,1,kernel_size=2,stride=1,padding=0)
        self.prelud1 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose1d(32,16,kernel_size=2,stride=1,padding=0)
        self.prelud2 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose1d(32,32,kernel_size=2,stride=1,padding=0)
        self.prelud3 = nn.PReLU()

    def predict_x(self,x):

        x1 = self.prelu1_1(self.conv1_1(x))
        x1 = self.prelu1_2(self.conv1_2(x1))
        x1 = F.max_pool1d(x1,kernel_size=2,stride=1)

        x2 = self.prelu2_1(self.conv2_1(x1))
        x2 = self.prelu2_2(self.conv2_2(x2))
        x2 = F.max_pool1d(x2,kernel_size=2,stride=1)

        x3 = self.prelu3_1(self.conv3_1(x2))
        x3 = self.prelu3_2(self.conv3_2(x3))
        x3 = F.max_pool1d(x3,kernel_size=2,stride=1)
 
        x4 = self.prelu3_3(self.conv3_3(x3))
        x4 = x4.view(x4.size(0), -1)

        x4 = self.prelufc4(self.fc4(x4))
        x5 = self.fc5(x4)

        z3 = self.prelubtl3(self.btl3(x3))  
        z2 = self.prelubtl2(self.btl2(x2))  
        z1 = self.prelubtl1(self.btl1(x1)) 

        j3 = self.btlu3(z3)
        j2 = self.btlu2(z2)
        j1 = self.btlu1(z1) # 16

        
        g2 = self.prelud3(self.deconv3(j3))
        g1 = self.prelud2(self.deconv2(j2+g2)) # 32
        g0 = self.prelud1(self.deconv1(j1+g1))

        return x5, x4, g0, [j3,j2,j1]

    def forward(self,x):

        x1 = self.prelu1_1(self.conv1_1(x))
        x1 = self.prelu1_2(self.conv1_2(x1))
        x1 = F.max_pool1d(x1,kernel_size=2,stride=1)
        # x1 = F.dropout(x1,p=0.15)

        x2 = self.prelu2_1(self.conv2_1(x1))
        x2 = self.prelu2_2(self.conv2_2(x2))
        x2 = F.max_pool1d(x2,kernel_size=2,stride=1)
        # x2 = F.dropout(x2,p=0.15)

        x3 = self.prelu3_1(self.conv3_1(x2))
        x3 = self.prelu3_2(self.conv3_2(x3))
        x3 = F.max_pool1d(x3,kernel_size=2,stride=1)
        # x3 = F.dropout(x3,p=0.15)
 
        x4 = self.prelu3_3(self.conv3_3(x3))
        x4 = x4.view(x4.size(0), -1)
        # print(x4.shape)

        x4 = self.prelufc4(self.fc4(x4))
        x5 = self.fc5(x4)

        z3 = self.prelubtl3(self.btl3(x3))  
        z2 = self.prelubtl2(self.btl2(x2))  
        z1 = self.prelubtl1(self.btl1(x1)) 

        j3 = self.btlu3(z3)
        j2 = self.btlu2(z2)
        j1 = self.btlu1(z1) # 16

        
        g2 = self.prelud3(self.deconv3(j3))
        g1 = self.prelud2(self.deconv2(j2+g2)) # 32
        g0 = self.prelud1(self.deconv1(j1+g1))

        return x5, x4, g0, [z3,z2,z1]

    def __lyr_freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def lyr_freeze(self):
        frz_lyr = [self.conv1_1, self.prelu1_1, self.conv1_2,
        self.prelu1_2, self.conv2_1, self.prelu2_1,
        self.conv2_2, self.prelu2_2, self.conv3_1,
        self.prelu3_1, self.conv3_2, self.prelu3_2,
        self.conv3_3, self.prelu3_3]
        for _ in frz_lyr:
            self.__lyr_freeze(_)