import torch
from torch import nn
import torch.nn.functional as F
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16], need_return_dic = True):
        super(densenet, self).__init__()
        self.need_return_dict = need_return_dic
        self.block1 = nn.Sequential(

            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        #self.ea = nn.Linear(1024,2)
    def build_results(self,x):
        return {
            "pred_logits":x,
        }
    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        #print(x.size())
        #a = self.ea(x)
        #print(a.size())
        x = self.classifier(x)
        #print(x.size())
        return self.build_results(x) if(self.need_return_dict) else x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = x.view(x.shape[0], -1, 4)
        return x
    
class ShareMLP(MLP):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ShareMLP, self).__init__(input_dim, hidden_dim, output_dim, num_layers)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.BN = nn.BatchNorm2d(hidden_dim)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x_src = x
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1:
                x = x + x_src
                x = self.BN(x)
        x = x.view(x.shape[0], -1, 4)
        return x
    
class DenseCoord(densenet):
    def __init__(self, in_channel, num_classes, num_queries = 25,growth_rate = 32, block_layers=[6, 12, 24, 16], need_return_dic = True):

        super(DenseCoord,self).__init__(in_channel, num_classes, growth_rate=growth_rate, block_layers=block_layers,
                                        need_return_dic = need_return_dic)
        self.num_classes = num_classes + 1
        self.class_embed = nn.Linear(1024, self.num_classes * num_queries)
        self.bbox_embed = MLP(1024, 1024, 4 * num_queries, 3)
        #self.bbox_embed = ShareMLP(1024, 1024, 4 * num_queries, 3)
        
    def build_results(self,x,y):
        return {
            "pred_logits":x,
            "pred_boxes":y,
        }
    def feature(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        return x
    def forward(self, x):
        feature_map = self.feature(x)

        class_feature = self.class_embed(feature_map)
        
        outputs_class = class_feature.view(class_feature.shape[0], -1, self.num_classes)    # one-hot

        outputs_coord = self.bbox_embed(feature_map).sigmoid()
        #print(outputs_class.shape)
        #print(outputs_coord.shape)
        return self.build_results(outputs_class, outputs_coord) if (self.need_return_dict) else [outputs_class,outputs_coord]



if __name__ == '__main__':
    model = DenseCoord(3, 8, num_queries = 25)
    image = torch.zeros((2,3,128,128))
    d = model(image)
    print(d['pred_boxes'].shape, d['pred_logits'].shape)
    # torch.Size([2, 25, 4]), torch.Size([2, 25, 8])