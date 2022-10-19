from dataloader import Dataload
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
import GPUtil
use_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(3407)
if(use_gpu):
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    if(len(deviceIDs) != 0):
        deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 1, maxMemory = 1, includeNan=False, excludeID=[], excludeUUID=[])
        print(deviceIDs)
        print("detect set :", deviceIDs)
        device = torch.device("cuda:"+str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)
class Train():
    def __init__(self, in_channles, out_channels, image_size = 128,is_show = True):
        self.in_channels = in_channles
        self.out_channels = out_channels
        self.image_size = image_size
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.create(is_show)
    
    def create(self, is_show):

        from model.DesNet import DenseCoord as Model
        self.model = Model(in_channel=self.in_channels, num_classes=self.out_channels, num_queries = 25)
        self.name = "dense121"

        self.costCross = torch.nn.CrossEntropyLoss()
        self.costL2= torch.nn.SmoothL1Loss()
        # self.cost = torch.nn.MSELoss()
        if(use_gpu):
            self.model = self.model.to(device)
            self.costCross = self.costCross.to(device)
            self.costL2 = self.costL2.to(device)
        if(is_show):
            summary(self.model, ( self.in_channels, self.image_size, self.image_size ))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.5, 0.999))
    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_acc = 0.0
        es = 0
        for epoch in range(n_epochs):
            start_time =datetime.datetime.now()
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-" * 10)
            epoch_train_acc, epoch_train_loss, coord_train_loss, class_train_loss = self.train(data_loader_train)
            epoch_test_acc, epoch_test_loss, coord_test_loss, class_test_loss = self.test(data_loader_test)

            self.history_acc.append(epoch_train_acc)
            self.history_loss.append(epoch_train_loss)
            self.history_test_acc.append(epoch_test_acc)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Coord:{:.4f}, Class:{:.4f}\nLoss is:{:.4f}, Test Accuracy is:{:.4f}%, Coord:{:.4f}, Class:{:.4f}\ncost time:{:.4f} min, EAT:{:.4f}".format(
                    epoch_train_loss,
                    epoch_train_acc * 100,
                    coord_train_loss, class_train_loss,
                    epoch_test_loss,
                    epoch_test_acc * 100,
                    coord_test_loss, class_test_loss,
                    (datetime.datetime.now() - start_time).seconds / 60,
                    (n_epochs - 1 - epoch) * (datetime.datetime.now() - start_time).seconds / 60,
                    )
            )
            if(epoch <= 4):
                continue
            if ((epoch_test_acc > 0.95 and epoch_test_acc > best_acc) or  (epoch_test_acc <= 0.95 and epoch_test_acc > best_acc + 1) ) :
                best_acc = epoch_test_acc
                es = 0
                self.save_parameter("./save_best/", "best")
            else:
                es += 1
                print("Counter {} of 10".format(es))

                if es > 9:
                    print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", epoch_test_acc, "...")
                    break
        self.save_history()
        self.save_parameter()


    def test(self,data_loader_test):
        self.model.eval()
        running_correct = 0
        running_loss =0
        test_index = 0
        coord_loss = 0
        class_loss = 0
        with torch.no_grad():
            for data in data_loader_test:
                X_test, y_test = data
                X_test, y_test = Variable(X_test).float(), Variable(y_test)
                if(use_gpu):
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)

                outputs = self.model(X_test)

                lossClass = self.costCross(outputs['pred_logits'].view(-1, self.out_channels + 1), y_test[:,:,0].long().view(-1))
                lossCoord = self.costL2(outputs["pred_boxes"].float(), y_test[:,:,1:].float())
                loss = lossClass + lossCoord
                pred = torch.argmax(outputs['pred_logits'].view(-1, self.out_channels + 1),1)
                label = y_test[:,:,0].long().view(-1)
                acc = torch.mean((pred == label).float())


                running_correct += acc.cpu().numpy()
                running_loss += loss.data.item()
                coord_loss += lossCoord.data.item()
                class_loss += lossClass.data.item()
                test_index += 1

        epoch_acc =  running_correct/(test_index+1)
        epoch_loss = running_loss/(test_index+1)
        return  epoch_acc, epoch_loss, coord_loss/(test_index+1), class_loss/(test_index+1)
            
    def train(self, data_loader_train):
        self.model.train()
        running_correct = 0
        train_index = 0
        running_loss = 0.0
        coord_loss = 0
        class_loss = 0
        for data in data_loader_train:
            X_train, y_train  = data
            X_train, y_train = Variable(X_train).float(), Variable(y_train)
            if(use_gpu):
                X_train = X_train.to(device)
                y_train = y_train.to(device)
            # print("训练中 train {}".format(X_train.shape))
            self.optimizer.zero_grad()

            outputs  = self.model(X_train)
            #print(outputs["pred_boxes"])
            lossClass = self.costCross(outputs['pred_logits'].view(-1, self.out_channels + 1).float(), y_train[:,:,0].long().view(-1))
            lossCoord = self.costL2(outputs["pred_boxes"].float(), y_train[:,:,1:].float())
            loss = lossClass + lossCoord
            #loss = loss.float()
            loss.backward()
            self.optimizer.step()

            pred = torch.argmax(outputs['pred_logits'].view(-1, self.out_channels + 1),1)
            label = y_train[:,:,0].long().view(-1)
            acc = torch.mean((pred == label).float())
            #acc = torch.mean([pred[i].cpu() == label[i] for i in range(len(pred))])
            #print(acc)
            coord_loss += lossCoord.data.item()
            class_loss += lossClass.data.item()

            running_loss += loss.data.item()
            running_correct += acc.cpu().data.item()
            train_index += 1

        epoch_train_acc = running_correct/train_index
        epoch_train_loss = running_loss/train_index
        return epoch_train_acc, epoch_train_loss, coord_loss/train_index, class_loss/train_index

    def predict_batch(self, image):

        if(type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if(len(image.size()) == 3 ):
            image.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            image = Variable(image).float()
            if(use_gpu):
                image = image.to(device)
            print(image.shape)
            output = self.model(image)
        # return output['pred_logits'],output['pred_boxes']直接返回output
        return output['pred_logits'],output['pred_boxes']

    def predict_each(self, image):
        if(type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if(len(image.size()) == 3 ):
            image.unsqueeze(1)
            # print(image)

        self.model.eval()
        with torch.no_grad():
            image = Variable(image).float()
            image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
            if(use_gpu):
                image = image.to(device)
            output = self.model(image)
            return  output

    def save_history(self, file_path = './save/'):
        file_path = file_path + self.name +"/"
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "loss_test_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()   
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close() 
    def save_parameter(self, file_path = './save/', name =None):
        file_path = file_path + self.name +"/"
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        if name ==None:
            file_path = file_path + "model_" +str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace("-","_").replace(".","_") + ".pkl"
        else:
            file_path = file_path + name + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)
    def load_parameter(self, file_path = './save/' ):
        self.model.load_state_dict(torch.load(file_path))

def trainTest():
    batch_size = 128
    image_size = 128
    data_path = r"E:\Dataset\training_set\train"

    All_dataloader = Dataload(r"E:\Dataset\training_set\train")

    train_size = int(len(All_dataloader.photo_set) * 0.8)
    validate_size = len(All_dataloader.photo_set) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])

    print("训练集大小: {} 测试集大小: {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    trainer = Train(3, 8, image_size, False)

    # trainer =  Train(3,25,image_size,False)
    # print(len(train_loader), len(test_loader))
    print("开始训练")
    trainer.train(train_loader)
    # trainer.train_and_test(100, train_loader, validate_loader)
    # trainer.test(validate_loader)

def perdit():
    trainer = Train(3, 8, 128, False)
    img = torch.ones([3, 128, 128])
    res = trainer.predict_each(img)
    print(res)

if __name__ == "__main__":
    batch_size = 128
    image_size = 128
    data_path = r"E:\Dataset\training_set\train"

    All_dataloader = Dataload(r"E:\Dataset\training_set\train")

    train_size = int(len(All_dataloader.photo_set) * 0.8)
    validate_size = len(All_dataloader.photo_set) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])

    print("训练集大小: {} 测试集大小: {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
    )

    trainer = Train(3, 8, image_size, False)
    trainer.train_and_test(20,train_loader,validate_loader)